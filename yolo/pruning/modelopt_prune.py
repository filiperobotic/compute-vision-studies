from __future__ import annotations

import io
import logging
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA, model_info

import modelopt.torch.prune as mtp


# =====================
# USER SETTINGS
# =====================
DATA_YAML = "./data.yaml"
IMG_SIZE = 640
BATCH = 16
DEVICE = 0

# Split epochs for baseline and pruning fine-tuning
EPOCHS_BASELINE = 100   # train original model
EPOCHS_PRUNED = 50      # fine-tune after pruning
BATCH = 16
DEVICE = 0

# Energy settings
POWER_W = 6.05      # measured power in Watts (for Joules/img)
E_MAC = 4.6e-12     # Joules per MAC (paper constant)
REPORT_TEST = True  # try to also print TEST mAP, if data.yaml has `test:`

# Models
ORIGINAL_MODEL_SRC = "yolov10x.pt"  # change to yolo11x.pt if you want

# Pruning
FLOPS_TARGET = "66%"
SEARCH_CKPT = "modelopt_fastnas_search_checkpoint.pth"


# =====================
# METRICS HELPERS
# =====================

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

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GFLOPs", txt)
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


def checkpoint_size_mb(path: str) -> float | None:
    if path and os.path.exists(path):
        return os.path.getsize(path) / 1e6
    return None


def report_metrics(
    yolo: YOLO,
    weights_path: str | None,
    title: str,
    imgsz: int,
    power_w: float,
    e_mac: float,
    data_yaml: str,
    report_test: bool,
):
    print("\n" + "=" * 70)
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

    # checkpoint size
    if weights_path:
        csz = checkpoint_size_mb(weights_path)
        if csz is not None:
            print(f"Checkpoint size (on disk): {csz:.1f} MB")

    # model summary
    try:
        yolo.info(imgsz=imgsz)
    except Exception:
        try:
            yolo.info()
        except Exception:
            pass


# =====================
# STAGE 1: BASELINE TRAINING (ORIGINAL MODEL)
# =====================

print("\n" + "#" * 70)
print("STAGE 1: Train baseline (original model)")
print("#" * 70)

baseline = YOLO(ORIGINAL_MODEL_SRC)
res_base = baseline.train(
    data=DATA_YAML,
    epochs=EPOCHS_BASELINE,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    exist_ok=True,
)

base_dir = getattr(res_base, "save_dir", None)
if base_dir is None:
    base_dir = Path("runs")

base_best = os.path.join(str(base_dir), "weights", "best.pt")
base_last = os.path.join(str(base_dir), "weights", "last.pt")
if not os.path.exists(base_best) and os.path.exists(base_last):
    base_best = base_last

print(f"[INFO] Baseline save_dir: {base_dir}")
print(f"[INFO] Baseline best checkpoint: {base_best}")

baseline_best_model = YOLO(base_best)
try:
    baseline_best_model.fuse()
except Exception:
    pass

report_metrics(
    baseline_best_model,
    base_best,
    "METRICS (BASELINE TRAINED - BEFORE PRUNING)",
    imgsz=IMG_SIZE,
    power_w=POWER_W,
    e_mac=E_MAC,
    data_yaml=DATA_YAML,
    report_test=REPORT_TEST,
)

# =====================
# PRUNED TRAINER
# =====================

model = YOLO(base_best)


class PrunedTrainer(model.task_map[model.task]["trainer"]):
    def _setup_train(self):
        super()._setup_train()

        def collect_func(batch):
            return self.preprocess_batch(batch)["img"]

        def score_func(m):
            m.eval()
            self.validator.args.save = False
            self.validator.args.plots = False
            self.validator.args.verbose = False
            self.validator.args.data = DATA_YAML
            metrics = self.validator(model=m)
            self.validator.args.save = self.args.save
            self.validator.args.plots = self.args.plots
            self.validator.args.verbose = self.args.verbose
            self.validator.args.data = self.args.data
            return metrics["fitness"]

        prune_constraints = {"flops": FLOPS_TARGET}

        # disable fusing checks
        self.model.is_fused = lambda: True

        # Remove stale search checkpoint (Torch 2.6 weights_only can break unpickling)
        if os.path.exists(SEARCH_CKPT):
            try:
                os.remove(SEARCH_CKPT)
                print(f"[ModelOpt] Removed stale search checkpoint: {SEARCH_CKPT}")
            except Exception as e:
                print(f"[WARN] Could not remove {SEARCH_CKPT}: {e}")

        self.model, _ = mtp.prune(
            model=self.model,
            mode="fastnas",
            constraints=prune_constraints,
            dummy_input=torch.randn(1, 3, self.args.imgsz, self.args.imgsz).to(self.device),
            config={
                "score_func": score_func,
                "checkpoint": SEARCH_CKPT,
                "data_loader": self.train_loader,
                "collect_func": collect_func,
                "max_iter_data_loader": 20,
            },
        )

        self.model.to(self.device)
        self.ema = ModelEMA(self.model)

        # Recreate optimizer and scheduler
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
        LOGGER.info("Applied pruning")


# =====================
# RUN
# =====================


# Train with pruning trainer
results = model.train(
    data=DATA_YAML,
    trainer=PrunedTrainer,
    epochs=EPOCHS_PRUNED,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    exist_ok=True,
)

# Resolve best.pt path
save_dir = getattr(results, "save_dir", None)
if save_dir is None:
    # best effort fallback
    save_dir = Path("runs")

best_pt = os.path.join(str(save_dir), "weights", "best.pt")
if not os.path.exists(best_pt):
    # Ultralytics default often uses runs/detect/train*
    # fall back to the latest run folder name in results if available
    best_pt = os.path.join(str(save_dir), "weights", "last.pt")

print(f"\n[INFO] Pruned training save_dir: {save_dir}")
print(f"[INFO] Pruned best checkpoint:  {best_pt}")

# PRUNED metrics
pruned_model = YOLO(best_pt)
report_metrics(
    pruned_model,
    best_pt,
    "METRICS (PRUNED MODEL - AFTER PRUNING)",
    imgsz=IMG_SIZE,
    power_w=POWER_W,
    e_mac=E_MAC,
    data_yaml=DATA_YAML,
    report_test=REPORT_TEST,
)