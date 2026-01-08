from ultralytics import YOLO
import os
import re
import io
import torch
from contextlib import redirect_stdout
import time
import logging

trained_model_path = '/home/pesquisador/pesquisa/filipe/compute-vision-studies/runs/train/yolov8m__oxford_tower_custom_train/weights/best.pt'

# Carrega o modelo treinado
model = YOLO(trained_model_path)

# Avalia o modelo utilizando o conjunto de teste
#results = model.val(data='data.yaml', split='test')
results = model.val(data='data.yaml', 
                    imgsz=640,
                    device=0, 
                    split='test')

# Exibe os principais resultados
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")

# -----------------------------
# Inference time (GPU) + Joules per image
# -----------------------------
# We estimate pure model forward time using the underlying torch module.
# This avoids dataset/IO/postprocess overhead from `model.val()`.
imgsz = 640
bs = 1
warmup_iters = 10
measure_iters = 100

# Power in Watts (set this based on your measurement setup)
# Example: export POWER_W=6.05
# power_w = float(os.getenv("POWER_W", "0"))
power_w = 6.05

# Prepare model + input
model.model.eval()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

x = torch.rand(bs, 3, imgsz, imgsz, device=device)

# Warmup
with torch.no_grad():
    for _ in range(warmup_iters):
        _ = model.model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

# Measure
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(measure_iters):
        _ = model.model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

t1 = time.perf_counter()

total_s = t1 - t0
sec_per_img = total_s / (measure_iters * bs)
ms_per_img = sec_per_img * 1e3

print(f"Infer time (forward-only, imgsz={imgsz}, bs={bs}): {ms_per_img:.3f} ms/img")

# Joules per image (measured power * measured time)
if power_w > 0:
    joules_per_img = power_w * sec_per_img
    print(f"Joules/img (POWER_W={power_w}): {joules_per_img:.6f} J  ({joules_per_img*1e3:.3f} mJ)")
else:
    print("Joules/img: set your measured power in Watts via env var POWER_W (e.g., export POWER_W=6.05)")

# Peak GPU memory during the timed forward (optional, only meaningful on CUDA)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model.model(x)
        torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak GPU memory (forward-only): {peak_mem_mb:.1f} MB")

# -----------------------------
# Model size (parameters)
# -----------------------------
# NOTE: `YOLO(...)` is a wrapper; the underlying torch module is `model.model`
n_params = sum(p.numel() for p in model.model.parameters())
print("Params:", n_params, "=>", n_params / 1e6, "M")

# -----------------------------
# GFLOPs extraction (automatic)
# -----------------------------

def get_gflops_ultralytics(yolo_model, imgsz: int = 640) -> float:
    """Try to extract GFLOPs from Ultralytics in a robust way.

    1) Prefer internal model_info() if available.
    2) Fallback: capture the printed summary/info and regex-parse the 'GFLOPs' value.
    """
    # 1) Try internal model_info helper (varies by Ultralytics version)
    try:
        from ultralytics.utils.torch_utils import model_info  # type: ignore
        # model_info returns (layers, params, gradients, flops) in some versions
        info = model_info(yolo_model.model, verbose=False, imgsz=imgsz)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            flops = info[3]
            # Some versions return FLOPs (not GFLOPs). Convert if large.
            if flops is not None:
                flops = float(flops)
                return flops / 1e9 if flops > 1e6 else flops
    except Exception:
        pass

    # 2) Fallback: Ultralytics often logs via LOGGER (not stdout). Capture logs and parse '... GFLOPs'.
    txt = ""

    # Try to capture Ultralytics logger output
    try:
        from ultralytics.utils import LOGGER  # type: ignore
        log_buf = io.StringIO()
        handler = logging.StreamHandler(log_buf)
        handler.setLevel(logging.INFO)

        # Temporarily attach handler
        LOGGER.addHandler(handler)
        try:
            yolo_model.info(imgsz=imgsz)
        except Exception:
            yolo_model.info()
        finally:
            LOGGER.removeHandler(handler)

        txt = log_buf.getvalue()
    except Exception:
        # Last resort: capture stdout (works on some versions)
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                yolo_model.info(imgsz=imgsz)
            except Exception:
                yolo_model.info()
        txt = buf.getvalue()

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GFLOPs", txt)
    if not m:
        raise RuntimeError(
            "Could not automatically extract GFLOPs from Ultralytics output. "
            "Try printing `model.info(imgsz=640)` and checking how it is logged in your environment."
        )
    return float(m.group(1))


gflops = get_gflops_ultralytics(model, imgsz=imgsz)
print(f"GFLOPs (imgsz={imgsz}): {gflops:.4f}")

# -----------------------------
# Analytical energy (paper style)
# -----------------------------
E_MAC = 4.6e-12  # J (paper, 45nm, FP32)

# Convention: 1 MAC = 2 FLOPs (common). Then MACs = FLOPs/2.
macs = gflops * 1e9 / 2
E_j = macs * E_MAC
print("MACs (G):", macs / 1e9)
print("Energia (J):", E_j)
print("Energia (mJ):", E_j * 1e3)


# path = "runs/train/.../weights/best.pt"
size_mb = os.path.getsize(trained_model_path) / 1e6
print(f"Checkpoint size: {size_mb:.1f} MB")