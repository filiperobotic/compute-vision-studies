#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pruning YOLO (Ultralytics) com Torch-Pruning no estilo do exemplo oficial do VainF,
adaptado para:
- NÃO deepcopy(YOLO) (evita erro de pickling do DataLoader iterator)
- Sanitizar "inference tensors" (PyTorch 2.6) antes do fine-tune
- Manter suas métricas: GFLOPs, MACs, energia analítica, tempo, Joules/img, peak GPU mem, tamanho do checkpoint
Base: VainF Torch-Pruning YOLOv8 pruning example.   [oai_citation:1‡GitHub](https://raw.githubusercontent.com/VainF/Torch-Pruning/refs/heads/master/examples/yolov8/yolov8_pruning.py)
"""

import argparse
import io
import logging
import math
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch_pruning as tp

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils.torch_utils import model_info


# ---------------------------
# MÉTRICAS (iguais ao seu estilo)
# ---------------------------

def get_gflops_ultralytics(yolo_model, imgsz: int = 640) -> float:
    # tenta model_info (rápido)
    try:
        info = model_info(yolo_model.model, verbose=False, imgsz=imgsz)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            flops = info[3]
            if flops is not None:
                flops = float(flops)
                return flops / 1e9 if flops > 1e6 else flops
    except Exception:
        pass

    # fallback: parse do .info()
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

    w_dtype = next(yolo_model.model.parameters()).dtype
    x = torch.rand(bs, 3, imgsz, imgsz, device=device, dtype=w_dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = yolo_model.model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

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
                   imgsz: int, power_w: float, e_mac: float):
    print("=" * 70)
    print(title)
    print("=" * 70)

    n_params = sum(p.numel() for p in yolo_model.model.parameters())
    print(f"Params: {n_params} => {n_params/1e6:.6f} M")

    gflops = get_gflops_ultralytics(yolo_model, imgsz=imgsz)
    print(f"GFLOPs (imgsz={imgsz}): {gflops:.4f}")

    # energia analítica
    macs = gflops * 1e9 / 2.0
    E_j = macs * e_mac
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
        size_mb = os.path.getsize(weights_path) / 1e6
        print(f"Checkpoint size: {size_mb:.1f} MB")

    print("=" * 70)
    print()


# ---------------------------
# Compat: C2f -> C2f_v2 (igual ideia do exemplo VainF)  [oai_citation:2‡GitHub](https://raw.githubusercontent.com/VainF/Torch-Pruning/refs/heads/master/examples/yolov8/yolov8_pruning.py)
# ---------------------------

def try_import_ultra_modules():
    mods = {}
    from ultralytics.nn.modules import Conv, Bottleneck, C2f
    mods["Conv"] = Conv
    mods["Bottleneck"] = Bottleneck
    mods["C2f"] = C2f
    try:
        from ultralytics.nn.modules import Detect
        mods["Detect"] = Detect
    except Exception:
        pass
    return mods


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, "add") and bottleneck.add


class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, mods=None):
        super().__init__()
        mods = mods or {}
        Conv = mods["Conv"]
        Bottleneck = mods["Bottleneck"]

        self.c = int(c2 * e)
        self.c_ = self.c
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m
    sd = c2f.state_dict()
    sd2 = c2f_v2.state_dict()

    old_w = sd["cv1.conv.weight"]
    half = old_w.shape[0] // 2
    sd2["cv0.conv.weight"] = old_w[:half]
    sd2["cv1.conv.weight"] = old_w[half:]

    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = sd[f"cv1.bn.{bn_key}"]
        sd2[f"cv0.bn.{bn_key}"] = old_bn[:half]
        sd2[f"cv1.bn.{bn_key}"] = old_bn[half:]

    for k, v in sd.items():
        if not k.startswith("cv1."):
            sd2[k] = v

    # importante: copiar metadados do grafo (i, f, type, np, n) se existirem
    for attr in ("i", "f", "type", "np", "n"):
        if hasattr(c2f, attr):
            setattr(c2f_v2, attr, getattr(c2f, attr))

    c2f_v2.load_state_dict(sd2, strict=False)


def replace_c2f_with_c2f_v2(module: nn.Module, mods):
    C2f = mods["C2f"]
    for name, child in list(module.named_children()):
        if isinstance(child, C2f):
            shortcut = False
            try:
                if hasattr(child, "m") and len(child.m) > 0:
                    shortcut = infer_shortcut(child.m[0])
            except Exception:
                pass

            c1 = child.cv1.conv.in_channels
            c2 = child.cv2.conv.out_channels
            n = len(child.m) if hasattr(child, "m") else 1
            e = float(child.c) / float(c2) if hasattr(child, "c") else 0.5
            g = child.m[0].cv2.conv.groups if (hasattr(child, "m") and len(child.m) > 0) else 1

            new_mod = C2f_v2(c1, c2, n=n, shortcut=shortcut, g=g, e=e, mods=mods)
            transfer_weights(child, new_mod)
            setattr(module, name, new_mod)
        else:
            replace_c2f_with_c2f_v2(child, mods)


# ---------------------------
# PyTorch 2.6: “inference tensors” -> parâmetros normais
# ---------------------------

def sanitize_inference_tensors(m: nn.Module) -> int:
    fixed = 0
    is_inf = getattr(torch, "is_inference", None)
    for mod in m.modules():
        for k, p in list(getattr(mod, "_parameters", {}).items()):
            if p is None:
                continue
            try:
                if is_inf is not None and is_inf(p):
                    mod._parameters[k] = nn.Parameter(p.detach().clone(), requires_grad=True)
                    fixed += 1
            except Exception:
                pass
    return fixed


# ---------------------------
# Trainer in-memory (não recarrega modelo)
# ---------------------------

def make_inmemory_trainer(base_trainer_cls, inmem_model: nn.Module):
    class InMemoryTrainer(base_trainer_cls):
        def __init__(self, cfg=None, overrides=None, _callbacks=None):
            self._inmem_model = inmem_model
            if cfg is None or cfg == "default":
                cfg = DEFAULT_CFG
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        def get_model(self, cfg=None, weights=None, verbose=True):
            return self._inmem_model

    return InMemoryTrainer


# ---------------------------
# LOOP principal (estilo VainF)
# ---------------------------

def prune_loop(args):
    mods = try_import_ultra_modules()

    model = YOLO(args.model)

    # IMPORTANTÍSSIMO: não chame fuse() no modelo que vai treinar
    model.model.train()
    replace_c2f_with_c2f_v2(model.model, mods)

    # baseline val com YOLO separado (evita deepcopy(YOLO))
    val_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=1,
        workers=args.workers,
        device=args.device,
        project=args.project,
    )
    baseline = YOLO(args.model)
    metric0 = baseline.val(**val_kwargs)
    init_map = float(metric0.box.map)
    init_map50 = float(metric0.box.map50)
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ops/params baseline
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz).to(model.device)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    print(f"[BASE] MACs={base_macs/1e9:.5f}G | Params={base_nparams/1e6:.5f}M | mAP={init_map:.5f} | mAP50={init_map50:.5f}")

    # métricas “do seu jeito” (usando um YOLO separado e fundido só para medir infer)
    try:
        m0 = YOLO(args.model)
        try:
            m0.fuse()
        except Exception:
            pass
        report_metrics(m0, args.model, "METRICS (ORIGINAL MODEL)", args.imgsz, args.power_w, args.e_mac)
        del m0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] Could not report baseline metrics: {e}")

    pruning_ratio = 1.0 - math.pow((1.0 - args.target_prune_rate), 1.0 / args.iterative_steps)

    for i in range(args.iterative_steps):
        keep = (1.0 - pruning_ratio) ** (i + 1)
        print(f"\n---------- [ITER {i+1}/{args.iterative_steps}] keep≈{keep:.3f} ----------")

        # reset (como o exemplo VainF sugere)  [oai_citation:3‡GitHub](https://raw.githubusercontent.com/VainF/Torch-Pruning/refs/heads/master/examples/yolov8/yolov8_pruning.py)
        try:
            model.model.criterion = None
        except Exception:
            pass

        model.model.train()
        for p in model.model.parameters():
            p.requires_grad = True

        # ignore heads
        ignored_layers = []
        if "Detect" in mods:
            for m in model.model.modules():
                if isinstance(m, mods["Detect"]):
                    ignored_layers.append(m)

        # importance (L2 magnitude)
        if hasattr(tp.importance, "GroupMagnitudeImportance"):
            importance = tp.importance.GroupMagnitudeImportance()
        elif hasattr(tp.importance, "MagnitudeImportance"):
            importance = tp.importance.MagnitudeImportance(p=2)
        else:
            raise RuntimeError("Nenhuma importance compatível: precisa de GroupMagnitudeImportance ou MagnitudeImportance")

        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs.to(model.device),
            importance=importance,
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=[],
        )

        print(f"[INFO] Using importance: {importance.__class__.__name__}")
        pruner.step()

        # pré-val
        metric_pre = model.val(**val_kwargs)
        pruned_map = float(metric_pre.box.map)
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model.model, example_inputs.to(model.device))
        speedup = float(base_macs) / float(pruned_macs)

        print(f"[PRUNED pre-FT] MACs={pruned_macs/1e9:.5f}G | Params={pruned_nparams/1e6:.5f}M | "
              f"mAP={pruned_map:.5f} | speedup~{speedup:.3f}x | param%={100.0*pruned_nparams/base_nparams:.2f}%")

        # SANITIZE (PyTorch 2.6)
        fixed = sanitize_inference_tensors(model.model)
        if fixed:
            print(f"[INFO] Sanitized {fixed} inference parameter(s) before fine-tune.")

        # fine-tune com Trainer “in-memory”
        base_trainer_cls = model.task_map[model.task]["trainer"]
        InMemTrainer = make_inmemory_trainer(base_trainer_cls, model.model)

        train_kwargs = dict(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            epochs=args.ft_epochs,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=f"tp_prune_step_{i+1:02d}_finetune",
        )

        with torch.enable_grad():
            model.train(trainer=InMemTrainer, **train_kwargs)

        # pós-val
        metric_post = model.val(**val_kwargs)
        cur_map = float(metric_post.box.map)
        cur_map50 = float(metric_post.box.map50)
        print(f"[POST-FT] mAP={cur_map:.5f} | mAP50={cur_map50:.5f} | drop={init_map-cur_map:.5f}")

        if (init_map - cur_map) > args.max_map_drop:
            print("[EARLY STOP] mAP drop exceeded threshold.")
            break

        del pruner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # salva
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / "pruned_final.pt"
    model.save(str(out_pt))
    print(f"[DONE] Saved pruned model to: {out_pt}")

    # métricas finais (fuse só no modelo de avaliação)
    try:
        mf = YOLO(str(out_pt))
        try:
            mf.fuse()
        except Exception:
            pass
        report_metrics(mf, str(out_pt), "METRICS (PRUNED FINAL MODEL)", args.imgsz, args.power_w, args.e_mac)
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

    p.add_argument("--iterative-steps", type=int, default=24)
    p.add_argument("--target-prune-rate", type=float, default=0.8)
    p.add_argument("--max-map-drop", type=float, default=0.15)
    p.add_argument("--ft-epochs", type=int, default=20)

    # energia/tempo
    p.add_argument("--power-w", type=float, default=6.05)
    p.add_argument("--e-mac", type=float, default=4.6e-12, help="J/MAC for analytical energy")

    p.add_argument("--out-dir", type=str, default="runs/prune_tp/out")
    return p


def main():
    args = build_argparser().parse_args()
    prune_loop(args)


if __name__ == "__main__":
    main()