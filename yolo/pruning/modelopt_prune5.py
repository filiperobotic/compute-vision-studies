#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train-from-scratch -> ModelOpt FastNAS prune -> fine-tune, with detailed metrics.

Stages:
  1) Train baseline from scratch (pretrained=True) from a model YAML.
  2) Apply ModelOpt FastNAS pruning to reach a FLOPs target (%).
  3) Fine-tune the pruned model.

Reported metrics (baseline + pruned):
- mAP50 / mAP50-95 on VAL split
- mAP50 / mAP50-95 on TEST split (if present)
- Params (M)
- GFLOPs (imgsz)
- MACs (G)
- Analytical energy (J/img, mJ/img) using E_MAC
- Forward-only inference time (ms/img)
- Joules/img using POWER_W
- Peak GPU memory (MB)
- Checkpoint size (MB)
- StateDict size (MB): native dtypes, fp32-cast, fp16-cast (fair size comparison)

Usage example:
  python -m yolo.pruning.modelopt_prune4 \
    --model yolov8m.yaml \
    --data data.yaml \
    --imgsz 640 \
    --epochs 100 \
    --ft-epochs 50 \
    --flops-target 66% \
    --power-w 6.05 \
    --e-mac 4.6e-12 \
    --report-test
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA, model_info

import modelopt.torch.prune as mtp


# ---------------------------
# PyTorch 2.6 safe unpickling (ModelOpt search checkpoint may contain defaultdict)
# ---------------------------
try:
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([defaultdict])
        print("[INFO] Added safe global: collections.defaultdict")
except Exception as e:
    print(f"[WARN] Could not add_safe_globals([defaultdict]): {e}")


# ---------------------------
# Metrics helpers
# ---------------------------

def extract_gflops(yolo: YOLO, imgsz: int) -> float:
    """Return GFLOPs from Ultralytics model_info; fallback to parsing yolo.info output."""
    try:
        if isinstance(getattr(yolo, "model", None), nn.Module):
            info = model_info(yolo.model, verbose=False, imgsz=imgsz)
            # (layers, params, gradients, flops)
            if isinstance(info, (list, tuple)) and len(info) >= 4:
                flops = info[3]
                if flops is not None:
                    flops = float(flops)
                    return flops / 1e9 if flops > 1e6 else flops
    except Exception:
        pass

    txt = ""
    try:
        from ultralytics.utils import LOGGER as _LOGGER
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.INFO)
        _LOGGER.addHandler(handler)
        try:
            yolo.info(imgsz=imgsz)
        except Exception:
            yolo.info()
        finally:
            _LOGGER.removeHandler(handler)
        txt = buf.getvalue()
    except Exception:
        txt = ""

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GFLOPs",e txt)
    if not m:
        raise RuntimeError("Could not extract GFLOPs from Ultralytics output.")
    return float(m.group(1))


def measure_forward_only(yolo: YOLO, imgsz: int, bs: int = 1, warmup: int = 20, iters: int = 200):
    """Forward-only time + peak VRAM. Only for torch nn.Module backends."""
    if not isinstance(getattr(yolo, "model", None), nn.Module):
        return None, None, None

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    yolo.model.to(device)
    yolo.model.eval()

    w_dtype = next(yolo.model.parameters()).dtype
    x = torch.rand(bs, 3, imgsz, imgsz, device=device, dtype=w_dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            _ = yolo.model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = yolo.model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    sec_per_img = (t1 - t0) / (iters * bs)
    ms_per_img = sec_per_img * 1e3

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

    return ms_per_img, sec_per_img, peak_mb


def state_dict_size_mb(yolo: YOLO, cast_dtype: torch.dtype | None = None) -> float | None:
    """Compute true tensor payload size from state_dict (no pickle/zip overhead).

    If cast_dtype is provided, floating tensors are virtually cast to that dtype for a fair comparison.
    """
    if not isinstance(getattr(yolo, "model", None), nn.Module):
        return None

    sd = yolo.model.state_dict()
    total_bytes = 0
    for _, t in sd.items():
        if not torch.is_tensor(t):
            continue
        tt = t
        if cast_dtype is not None and tt.is_floating_point():
            # virtual cast for size estimate
            if cast_dtype == torch.float16:
                bpe = 2
            elif cast_dtype == torch.float32:
                bpe = 4
            else:
                bpe = torch.tensor([], dtype=cast_dtype).element_size()
            total_bytes += tt.numel() * bpe
        else:
            total_bytes += tt.numel() * tt.element_size()
    return total_bytes / 1e6


def val_maps(yolo: YOLO, data_yaml: str, imgsz: int, split: str) -> tuple[float | None, float | None]:
    """Return (mAP50, mAP50-95) for the given split. If split missing, returns (None, None)."""
    try:
        res = yolo.val(data=data_yaml, imgsz=imgsz, split=split, verbose=False, rect=False)
        m50 = float(res.box.map50)
        mmap = float(res.box.map)
        return m50, mmap
    except Exception as e:
        print(f"[WARN] Could not run .val(split='{split}'): {e}")
        return None, None


def report_metrics(
    yolo: YOLO,
    weights_path: str,
    title: str,
    imgsz: int,
    power_w: float,
    e_mac: float,
    data_yaml: str,
    report_test: bool,
):
    print("=" * 70)
    print(title)
    print("=" * 70)

    # mAP val/test
    m50_v, mmap_v = val_maps(yolo, data_yaml=data_yaml, imgsz=imgsz, split="val")
    if m50_v is not None:
        print(f"VAL mAP50: {m50_v:.4f}")
        print(f"VAL mAP50-95: {mmap_v:.4f}")

    if report_test:
        m50_t, mmap_t = val_maps(yolo, data_yaml=data_yaml, imgsz=imgsz, split="test")
        if m50_t is not None:
            print(f"TEST mAP50: {m50_t:.4f}")
            print(f"TEST mAP50-95: {mmap_t:.4f}")

    # params and dtype snapshot + fair sizes
    try:
        if isinstance(getattr(yolo, "model", None), nn.Module):
            params = list(yolo.model.parameters())
            n_params = sum(p.numel() for p in params)
            dtypes = [p.dtype for p in params]
            fp16 = sum(1 for dt in dtypes if dt == torch.float16)
            fp32 = sum(1 for dt in dtypes if dt == torch.float32)
            print(f"Params: {n_params} => {n_params/1e6:.6f} M")
            print(f"Param dtypes: fp16={fp16}, fp32={fp32}, other={len(dtypes)-fp16-fp32}")

            native_mb = state_dict_size_mb(yolo, cast_dtype=None)
            fp32_mb = state_dict_size_mb(yolo, cast_dtype=torch.float32)
            fp16_mb = state_dict_size_mb(yolo, cast_dtype=torch.float16)
            if native_mb is not None:
                print(f"StateDict size (native dtypes): {native_mb:.1f} MB")
            if fp32_mb is not None:
                print(f"StateDict size (fp32 cast): {fp32_mb:.1f} MB")
            if fp16_mb is not None:
                print(f"StateDict size (fp16 cast): {fp16_mb:.1f} MB")
    except Exception as e:
        print(f"[WARN] Could not count params / state_dict sizes: {e}")

    # gflops / macs / analytical energy
    try:
        gflops = extract_gflops(yolo, imgsz=imgsz)
        print(f"GFLOPs (imgsz={imgsz}): {gflops:.4f}")
        macs = gflops * 1e9 / 2.0
        e_j = macs * e_mac
        print(f"MACs (G): {macs/1e9:.4f}")
        print(f"Analytical energy (J/img): {e_j:.6f} J  ({e_j*1e3:.3f} mJ)")
    except Exception as e:
        print(f"[WARN] Could not compute GFLOPs/MACs/analytical energy: {e}")

    # forward-only time / joules / vram
    ms_img, sec_img, peak_mb = measure_forward_only(yolo, imgsz=imgsz)
    if ms_img is not None:
        print(f"Infer time (forward-only, imgsz={imgsz}, bs=1): {ms_img:.3f} ms/img")
        if power_w and sec_img is not None:
            j_img = power_w * sec_img
            print(f"Joules/img (POWER_W={power_w}): {j_img:.6f} J  ({j_img*1e3:.3f} mJ)")
        if peak_mb is not None:
            print(f"Peak GPU memory (forward-only): {peak_mb:.1f} MB")
    else:
        print("[INFO] Forward-only timing skipped (non-torch backend).")

    # file size on disk
    if weights_path and os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / 1e6
        print(f"Checkpoint size (on disk): {size_mb:.1f} MB")

    print("=" * 70)
    print()


# ---------------------------
# Trainer factory: applies ModelOpt prune before training loop starts
# ---------------------------

def make_pruned_trainer(base_trainer_cls, data_yaml: str, flops_target: str):
    class PrunedTrainer(base_trainer_cls):
        def _setup_train(self):
            super()._setup_train()

            def collect_func(batch):
                return self.preprocess_batch(batch)["img"]

            def score_func(model: nn.Module):
                model.eval()
                # Disable logs
                self.validator.args.save = False
                self.validator.args.plots = False
                self.validator.args.verbose = False
                self.validator.args.data = data_yaml
                metrics = self.validator(model=model)
                # Restore
                self.validator.args.save = self.args.save
                self.validator.args.plots = self.args.plots
                self.validator.args.verbose = self.args.verbose
                self.validator.args.data = self.args.data
                return metrics["fitness"]

            # Versioned checkpoint (avoid stale/incompatible file)
            ckpt_name = f"modelopt_fastnas_search_checkpoint_torch{torch.__version__.split('+')[0]}.pth"
            if os.path.exists(ckpt_name):
                print(f"[ModelOpt] Removing existing search checkpoint: {ckpt_name}")
                try:
                    os.remove(ckpt_name)
                except Exception as e:
                    print(f"[WARN] Could not remove {ckpt_name}: {e}")

            prune_constraints = {"flops": flops_target}

            # Ultralytics fuse guard
            self.model.is_fused = lambda: True

            self.model, _prune_res = mtp.prune(
                model=self.model,
                mode="fastnas",
                constraints=prune_constraints,
                dummy_input=torch.randn(1, 3, self.args.imgsz, self.args.imgsz).to(self.device),
                config={
                    "score_func": score_func,
                    "checkpoint": ckpt_name,
                    "data_loader": self.train_loader,
                    "collect_func": collect_func,
                    "max_iter_data_loader": 20,
                },
            )

            self.model.to(self.device)
            self.ema = ModelEMA(self.model)

            # Recreate optimizer/scheduler
            weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
            iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
            self.optimizer = self.build_optimizer(
                model=self.model,
                name=self.args.optimizer,
                lr=self.args.lr0,
                momentum=self.args.momentum,
                decay=weight_decay,
                iterations=iterations,
            )
            self._setup_scheduler()
            LOGGER.info("Applied ModelOpt FastNAS pruning")

    return PrunedTrainer


# ---------------------------
# Stages
# ---------------------------

def train_baseline(args) -> str:
    print("\n==============================")
    print("STAGE 1: Train baseline from scratch")
    print("==============================\n")

    y = YOLO(args.model)
    results = y.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name_baseline,
        pretrained=True,
        exist_ok=True,
        warmup_epochs=0,
    )

    save_dir = getattr(results, "save_dir", None) or (Path(args.project) / args.name_baseline)
    best_pt = os.path.join(str(save_dir), "weights", "best.pt")
    last_pt = os.path.join(str(save_dir), "weights", "last.pt")
    if not os.path.exists(best_pt) and os.path.exists(last_pt):
        best_pt = last_pt

    print(f"[INFO] Baseline save_dir: {save_dir}")
    print(f"[INFO] Baseline best_pt:  {best_pt}")

    y_eval = YOLO(best_pt)
    try:
        y_eval.fuse()
    except Exception:
        pass

    report_metrics(
        y_eval,
        best_pt,
        "METRICS (BASELINE BEST)",
        imgsz=args.imgsz,
        power_w=args.power_w,
        e_mac=args.e_mac,
        data_yaml=args.data,
        report_test=args.report_test,
    )

    return best_pt


def prune_and_finetune(args, baseline_best: str) -> str:
    print("\n==============================")
    print("STAGE 2: Prune (FastNAS) + fine-tune")
    print("==============================\n")

    y = YOLO(baseline_best)
    base_trainer_cls = y.task_map[y.task]["trainer"]
    PrunedTrainer = make_pruned_trainer(base_trainer_cls, data_yaml=args.data, flops_target=args.flops_target)

    results = y.train(
        data=args.data,
        epochs=args.ft_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name_pruned,
        trainer=PrunedTrainer,
        exist_ok=True,
        warmup_epochs=0,
    )

    save_dir = getattr(results, "save_dir", None) or (Path(args.project) / args.name_pruned)
    best_pt = os.path.join(str(save_dir), "weights", "best.pt")
    last_pt = os.path.join(str(save_dir), "weights", "last.pt")
    if not os.path.exists(best_pt) and os.path.exists(last_pt):
        best_pt = last_pt

    print(f"[INFO] Pruned save_dir: {save_dir}")
    print(f"[INFO] Pruned best_pt:  {best_pt}")

    y_eval = YOLO(best_pt)
    try:
        y_eval.fuse()
    except Exception:
        pass

    report_metrics(
        y_eval,
        best_pt,
        "METRICS (PRUNED BEST)",
        imgsz=args.imgsz,
        power_w=args.power_w,
        e_mac=args.e_mac,
        data_yaml=args.data,
        report_test=args.report_test,
    )

    return best_pt


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        type=str,
        default="yolov8m.yaml",
        help="Model config YAML to train from scratch (e.g., yolov8m.yaml, yolov11x.yaml)",
    )
    p.add_argument("--data", type=str, required=True, help="data.yaml")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100, help="baseline epochs (scratch)")
    p.add_argument("--ft-epochs", type=int, default=50, help="fine-tune epochs after pruning")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--flops-target", type=str, default="66%", help="FastNAS FLOPs target (e.g., 66%)")

    p.add_argument("--power-w", type=float, default=6.05, help="Measured power in Watts")
    p.add_argument("--e-mac", type=float, default=4.6e-12, help="Joules per MAC (paper constant)")

    p.add_argument("--project", type=str, default="runs/modelopt")
    p.add_argument("--name-baseline", type=str, default="baseline_scratch")
    p.add_argument("--name-pruned", type=str, default="pruned_finetune")

    p.add_argument("--report-test", action="store_true", help="Also compute and print mAP on split='test'")

    return p.parse_args()


def main():
    args = parse_args()
    baseline_best = train_baseline(args)
    prune_and_finetune(args, baseline_best)


if __name__ == "__main__":
    main()