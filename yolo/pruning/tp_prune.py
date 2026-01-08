#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import io
import re
import time
import logging
from copy import deepcopy
from pathlib import Path
from contextlib import redirect_stdout

import torch
import torch.nn as nn

import torch_pruning as tp
from ultralytics import YOLO

# ---------------------------
# Metrics helpers (iguais ao seu estilo)
# ---------------------------

def get_gflops_ultralytics(yolo_model, imgsz: int = 640) -> float:
    """
    Extrai GFLOPs do Ultralytics (tentando model_info primeiro e fallback para parsing do .info()).
    """
    try:
        from ultralytics.utils.torch_utils import model_info  # type: ignore
        info = model_info(yolo_model.model, verbose=False, imgsz=imgsz)
        # Em muitas versões: info = (layers, params, gradients, flops)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            flops = info[3]
            if flops is not None:
                flops = float(flops)
                return flops / 1e9 if flops > 1e6 else flops
    except Exception:
        pass

    # Fallback: interceptar logs/print do yolo_model.info()
    txt = ""
    try:
        from ultralytics.utils import LOGGER  # type: ignore
        log_buf = io.StringIO()
        handler = logging.StreamHandler(log_buf)
        handler.setLevel(logging.INFO)
        LOGGER.addHandler(handler)
        try:
            yolo_model.info(imgsz=imgsz)
        except Exception:
            yolo_model.info()
        finally:
            LOGGER.removeHandler(handler)
        txt = log_buf.getvalue()
    except Exception:
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                yolo_model.info(imgsz=imgsz)
            except Exception:
                yolo_model.info()
        txt = buf.getvalue()

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GFLOPs", txt)
    if not m:
        raise RuntimeError("Could not extract GFLOPs from Ultralytics output.")
    return float(m.group(1))


def measure_forward_time_and_mem(yolo_model, imgsz: int = 640, bs: int = 1,
                                 warmup_iters: int = 20, measure_iters: int = 200):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    yolo_model.model.to(device)
    yolo_model.model.eval()

    # use dtype do modelo
    w_dtype = next(yolo_model.model.parameters()).dtype
    x = torch.rand(bs, 3, imgsz, imgsz, device=device, dtype=w_dtype)

    # reset peak para medir só inferência
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # warmup (compila kernels/caches)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = yolo_model.model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # tempo
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = yolo_model.model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    sec_per_img = (t1 - t0) / (measure_iters * bs)
    ms_per_img = sec_per_img * 1e3

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    return ms_per_img, sec_per_img, peak_mem_mb


def report_metrics(yolo_model, weights_path: str, title: str,
                   imgsz: int, power_w: float):
    print("=" * 70)
    print(title)
    print("=" * 70)

    n_params = sum(p.numel() for p in yolo_model.model.parameters())
    print(f"Params: {n_params} => {n_params/1e6:.6f} M")

    gflops = get_gflops_ultralytics(yolo_model, imgsz=imgsz)
    print(f"GFLOPs (imgsz={imgsz}): {gflops:.4f}")

    # energia analítica (mesma lógica que você usou)
    E_MAC = 4.6e-12  # J por MAC (do paper que você estava usando)
    macs = gflops * 1e9 / 2.0
    E_j = macs * E_MAC
    print(f"MACs (G): {macs/1e9:.4f}")
    print(f"Analytical energy (J/img): {E_j:.6f} J  ({E_j*1e3:.3f} mJ)")

    ms_per_img, sec_per_img, peak_mem_mb = measure_forward_time_and_mem(yolo_model, imgsz=imgsz)
    print(f"Infer time (forward-only, imgsz={imgsz}, bs=1): {ms_per_img:.3f} ms/img")

    if power_w > 0:
        joules_per_img = power_w * sec_per_img
        print(f"Joules/img (POWER_W={power_w}): {joules_per_img:.6f} J  ({joules_per_img*1e3:.3f} mJ)")

    if peak_mem_mb is not None:
        print(f"Peak GPU memory (forward-only): {peak_mem_mb:.1f} MB")

    if weights_path and os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / 1e6  # MB decimal
        print(f"Checkpoint size: {size_mb:.1f} MB")

    print("=" * 70)
    print()


# ---------------------------
# C2f -> C2f_v2 (para compatibilidade do graph parser)
# Baseado no tutorial: split do cv1 em dois ramos (cv0 e cv1) e concatena.  [oai_citation:1‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
# ---------------------------

def try_import_ultra_modules():
    mods = {}
    from ultralytics.nn.modules import Conv, Bottleneck, C2f
    mods["Conv"] = Conv
    mods["Bottleneck"] = Bottleneck
    mods["C2f"] = C2f

    # opcionais (variam por versão)
    try:
        from ultralytics.nn.modules import Detect
        mods["Detect"] = Detect
    except Exception:
        pass
    try:
        from ultralytics.nn.modules import v10Detect
        mods["v10Detect"] = v10Detect
    except Exception:
        pass
    try:
        from ultralytics.nn.modules import Attention
        mods["Attention"] = Attention
    except Exception:
        pass
    try:
        from ultralytics.nn.modules import PSA
        mods["PSA"] = PSA
    except Exception:
        pass
    try:
        from ultralytics.nn.modules import CIB, RepVGGDW
        mods["CIB"] = CIB
        mods["RepVGGDW"] = RepVGGDW
    except Exception:
        pass

    return mods


def infer_shortcut_and_variants(bottleneck_module, mods):
    """
    Heurística do tutorial para decidir shortcut / CIB / large-kernel.
    Se não conseguir, assume shortcut=False.
    """
    Bottleneck = mods.get("Bottleneck", None)
    RepVGGDW = mods.get("RepVGGDW", None)

    try:
        if Bottleneck is not None and isinstance(bottleneck_module, Bottleneck):
            c1 = bottleneck_module.cv1.conv.in_channels
            c2 = bottleneck_module.cv2.conv.out_channels
            return (c1 == c2 and getattr(bottleneck_module, "add", False)), False, False

        # Caso CIB/variante (depende da sua versão)
        add = getattr(bottleneck_module, "add", False)
        lk = False
        if RepVGGDW is not None:
            # tenta detectar large kernel
            try:
                lk = any(isinstance(mod, RepVGGDW) for mod in bottleneck_module.cv1)
            except Exception:
                lk = False
        return add, True, lk
    except Exception:
        return False, False, False


class C2f_v2(nn.Module):
    """
    Versão pruning-friendly do C2f: separa o primeiro conv em dois ramos (cv0 e cv1),
    depois concatena [cv0(x), cv1(x), m(...), ...] e aplica cv2.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, mods=None, is_CIB=False, lk=False):
        super().__init__()
        mods = mods or {}
        Conv = mods["Conv"]
        Bottleneck = mods["Bottleneck"]
        CIB = mods.get("CIB", None)

        self.c = int(c2 * e)
        # Keep Ultralytics-compatible attribute name used by some utilities
        self.c_ = self.c
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        if (not is_CIB) or (CIB is None):
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        else:
            # fallback para CIB se existir
            self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_c2f_weights(c2f, c2f_v2):
    """
    Transfere pesos do C2f original para C2f_v2 (mesma ideia do tutorial).  [oai_citation:2‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
    """
    # manter cv2 e m
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    sd = c2f.state_dict()
    sd2 = c2f_v2.state_dict()

    # cv1 do C2f original é split em dois convs (cv0 e cv1) na v2
    old_w = sd["cv1.conv.weight"]
    half = old_w.shape[0] // 2
    sd2["cv0.conv.weight"] = old_w[:half]
    sd2["cv1.conv.weight"] = old_w[half:]

    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = sd[f"cv1.bn.{bn_key}"]
        sd2[f"cv0.bn.{bn_key}"] = old_bn[:half]
        sd2[f"cv1.bn.{bn_key}"] = old_bn[half:]

    # resto
    for k, v in sd.items():
        if not k.startswith("cv1."):
            sd2[k] = v

    c2f_v2.load_state_dict(sd2, strict=False)


def replace_c2f_with_c2f_v2(module: nn.Module, mods):
    C2f = mods["C2f"]
    for name, child in list(module.named_children()):
        if isinstance(child, C2f):
            # tenta inferir shortcut / CIB / lk a partir do primeiro bloco interno
            shortcut, is_CIB, lk = False, False, False
            try:
                if hasattr(child, "m") and len(child.m) > 0:
                    shortcut, is_CIB, lk = infer_shortcut_and_variants(child.m[0], mods)
            except Exception:
                pass

            c1 = child.cv1.conv.in_channels
            c2 = child.cv2.conv.out_channels
            n = len(child.m) if hasattr(child, "m") else 1
            e = float(child.c) / float(c2) if hasattr(child, "c") else 0.5
            g = child.m[0].cv2.conv.groups if (hasattr(child, "m") and len(child.m) > 0 and hasattr(child.m[0], "cv2")) else 1

            new_mod = C2f_v2(c1, c2, n=n, shortcut=shortcut, g=g, e=e, mods=mods, is_CIB=is_CIB, lk=lk)
            transfer_c2f_weights(child, new_mod)

            # Ultralytics graph metadata (required by tasks.py _predict_once)
            for attr in ("i", "f", "type", "np", "n"):
                if hasattr(child, attr):
                    setattr(new_mod, attr, getattr(child, attr))

            setattr(module, name, new_mod)
        else:
            replace_c2f_with_c2f_v2(child, mods)


# ---------------------------
# Trainer in-memory (para fine-tune sem tentar “recriar” o modelo)
# ---------------------------

def make_inmemory_trainer(base_trainer_cls, inmem_model: nn.Module):
    """
    Cria uma Trainer class que devolve o modelo já prunado em memória, em vez de reconstruir/attempt_load.
    Isso evita quebra por mudança de arquitetura após pruning.
    """
    class InMemoryTrainer(base_trainer_cls):
        def __init__(self, cfg=None, overrides=None, _callbacks=None):
            self._inmem_model = inmem_model
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        def get_model(self, cfg=None, weights=None, verbose=True):
            # usa o modelo já prunado
            return self._inmem_model

    return InMemoryTrainer


# ---------------------------
# Pruning loop (iterative)
# ---------------------------

def iterative_prune(args):
    mods = try_import_ultra_modules()

    # carrega modelo
    model = YOLO(args.model)

    # prepara cfg para treinos/vals
    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.ft_epochs,    # epochs de fine-tune por iteração
        workers=args.workers,
        device=args.device,
        project=args.project,
    )

    val_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=1,
        workers=args.workers,
        device=args.device,
        project=args.project,
    )

    # troca C2f -> C2f_v2 (crucial para torch-pruning no YOLOv10, segundo o tutorial)  [oai_citation:3‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
    model.model.train()
    replace_c2f_with_c2f_v2(model.model, mods)

    # baseline (val)
    baseline = deepcopy(model)
    metric0 = baseline.val(**val_kwargs)
    init_map = float(metric0.box.map)
    init_map50 = float(metric0.box.map50)

    # baseline ops/params
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz).to(model.device)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    print(f"[BASE] MACs={base_macs/1e9:.5f}G | Params={base_nparams/1e6:.5f}M | mAP={init_map:.5f} | mAP50={init_map50:.5f}")

    # métricas “do jeito que você quer”
    try:
        model_eval = YOLO(args.model)
        try:
            model_eval.fuse()
        except Exception:
            pass
        report_metrics(model_eval, args.model, "METRICS (ORIGINAL MODEL)", args.imgsz, args.power_w)
        del model_eval
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] Could not report baseline metrics: {e}")

    # pruning_ratio por iteração (mesma fórmula do tutorial)  [oai_citation:4‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
    pruning_ratio = 1.0 - math.pow((1.0 - args.target_prune_rate), 1.0 / args.iterative_steps)

    # histórico (opcional)
    macs_list = [base_macs]
    nparams_list = [100.0]
    map_list = [init_map]
    map50_list = [init_map50]
    pruned_map_list = [init_map]

    # loop
    for i in range(args.iterative_steps):
        target_keep = (1.0 - pruning_ratio) ** (i + 1)
        print(f"\n---------- [ITER {i+1}/{args.iterative_steps}] keep≈{target_keep:.3f} ----------")

        # reset criterion como o tutorial sugere (evita interferência interna do Ultralytics)  [oai_citation:5‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
        try:
            model.model.criterion = None
        except Exception:
            pass

        model.model.train()
        for p in model.model.parameters():
            p.requires_grad = True

        # ignored layers (detect heads / attention)  [oai_citation:6‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
        ignored_layers = []
        for m in model.model.modules():
            if ("v10Detect" in mods and isinstance(m, mods["v10Detect"])) or \
               ("Detect" in mods and isinstance(m, mods["Detect"])) or \
               ("Attention" in mods and isinstance(m, mods["Attention"])) or \
               ("PSA" in mods and isinstance(m, mods["PSA"])):
                ignored_layers.append(m)

        # importance
        # importance (torch-pruning API muda entre versões)
        if hasattr(tp.importance, "GroupNormImportance"):
            importance = tp.importance.GroupNormImportance()
        elif hasattr(tp.importance, "GroupTaylorImportance"):
            importance = tp.importance.GroupTaylorImportance()
        elif hasattr(tp.importance, "MagnitudeImportance"):
            importance = tp.importance.MagnitudeImportance(p=2)  # L2 magnitude
        else:
            raise RuntimeError(f"Nenhuma importance compatível encontrada. Disponíveis: {dir(tp.importance)}")

        # pruner (GroupNormPruner)  [oai_citation:7‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs.to(model.device),
            importance=importance,
            iterative_steps=1,            # 1 por iteração do seu loop
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=[],      # mantém vazio como no tutorial
        )

        # aplica pruning
        pruner.step()

        # valida “pré-finetune”
        pre_val = deepcopy(model)
        metric_pre = pre_val.val(**val_kwargs)
        pruned_map = float(metric_pre.box.map)
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model.model, example_inputs.to(model.device))
        speedup = float(macs_list[0]) / float(pruned_macs)

        print(f"[PRUNED pre-FT] MACs={pruned_macs/1e9:.5f}G | Params={pruned_nparams/1e6:.5f}M | "
              f"mAP={pruned_map:.5f} | speedup~{speedup:.3f}x | param%={100.0*pruned_nparams/base_nparams:.2f}%")

        # fine-tune usando Trainer que NÃO recarrega weights/arquitetura
        base_trainer_cls = model.task_map[model.task]["trainer"]
        InMemTrainer = make_inmemory_trainer(base_trainer_cls, model.model)

        # roda treino curto para recuperar mAP
        train_kwargs_step = dict(train_kwargs)
        train_kwargs_step["name"] = f"tp_prune_step_{i+1:02d}_finetune"
        model.train(trainer=InMemTrainer, **train_kwargs_step)

        # pós-finetune: valida
        post_val = deepcopy(model)
        metric_post = post_val.val(**val_kwargs)
        cur_map = float(metric_post.box.map)
        cur_map50 = float(metric_post.box.map50)

        print(f"[POST-FT] mAP={cur_map:.5f} | mAP50={cur_map50:.5f} | drop={init_map-cur_map:.5f}")

        macs_list.append(pruned_macs)
        nparams_list.append(100.0 * pruned_nparams / base_nparams)
        pruned_map_list.append(pruned_map)
        map_list.append(cur_map)
        map50_list.append(cur_map50)

        # early stop (max mAP drop)  [oai_citation:8‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
        if (init_map - cur_map) > args.max_map_drop:
            print("[EARLY STOP] mAP drop exceeded threshold.")
            break

        del pruner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # salva modelo final (state_dict do YOLO após pruning)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / "pruned_final.pt"

    # Salvar como Ultralytics checkpoint simples: reaproveita o .save do YOLO
    # (melhor compatibilidade para depois carregar com YOLO(out_pt))
    model.save(str(out_pt))
    print(f"[DONE] Saved pruned model to: {out_pt}")

    # métricas finais (fused para infer timing)
    try:
        model_eval_final = YOLO(str(out_pt))
        try:
            model_eval_final.fuse()
        except Exception:
            pass
        report_metrics(model_eval_final, str(out_pt), "METRICS (PRUNED FINAL MODEL)", args.imgsz, args.power_w)
    except Exception as e:
        print(f"[WARN] Could not report final metrics: {e}")

    return str(out_pt)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="path to pretrained .pt (e.g., best.pt)")
    p.add_argument("--data", type=str, required=True, help="data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None, help="e.g. 0 or 'cpu'")
    p.add_argument("--project", type=str, default="runs/prune_tp", help="ultralytics project dir")

    p.add_argument("--importance", type=str, default="L2", choices=["L2"])
    p.add_argument("--iterative-steps", type=int, default=24)        # tutorial default  [oai_citation:9‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
    p.add_argument("--target-prune-rate", type=float, default=0.8)   # tutorial default  [oai_citation:10‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
    p.add_argument("--max-map-drop", type=float, default=0.15)       # tutorial default  [oai_citation:11‡Freedium](https://freedium-mirror.cfd/https%3A//medium.com/%40antonioconsiglio/how-to-prune-yolov10-with-iterative-pruning-and-torch-pruning-library-full-guide-0cded392389e)
    p.add_argument("--ft-epochs", type=int, default=20, help="fine-tune epochs per pruning step")
    p.add_argument("--power-w", type=float, default=6.05)

    p.add_argument("--out-dir", type=str, default="runs/prune_tp/out")
    return p


def main():
    args = build_argparser().parse_args()
    iterative_prune(args)


if __name__ == "__main__":
    main()