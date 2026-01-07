from ultralytics import YOLO
import torch
import re
import io
import time
import logging
from contextlib import redirect_stdout
import torch.nn as nn
import torch_pruning as tp
print(f"[torch-pruning] version: {getattr(tp, '__version__', 'unknown')}")
from utils import metrics
import os

# root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# oxford_town_base_model = os.path.join(root, 'yolo', 'oxford_base_models', 'trained_by_yolo11x.pt')
oxford_town_base_model = trained_model_path = '/home/pesquisador/pesquisa/filipe/compute-vision-studies/runs/train/yolo11x__oxford_tower_custom_train/weights/best.pt'

# Verifica se o arquivo do modelo existe antes de tentar usá-lo
if not os.path.exists(oxford_town_base_model):
    print(f"Modelo não encontrado em: {oxford_town_base_model}")
    print("Coloque o arquivo do modelo na pasta 'oxford_base_models' ou ajuste o caminho no script.")
    raise SystemExit(1)

print("=" * 60)
print("TAMANHO DO MODELO ORIGINAL")
print("=" * 60)
original_size = metrics.get_model_size(oxford_town_base_model)
print(f"Tamanho do arquivo em disco: {original_size:.2f} MB")

model = YOLO(oxford_town_base_model)
model_nn = model.model

# Fuse (Conv+BN) for inference-style graph (fair comparison)
try:
    model.fuse()
except Exception:
    try:
        model.model.fuse()
    except Exception:
        pass
model_nn = model.model
model_nn.eval()

# -----------------------------
# Metrics helpers (same as yolo/standard/test_yolo.py)
# -----------------------------

def get_gflops_ultralytics(yolo_model, imgsz: int = 640) -> float:
    """Try to extract GFLOPs from Ultralytics in a robust way.

    1) Prefer internal model_info() if available.
    2) Fallback: capture LOGGER/stdout and regex-parse the 'GFLOPs' value.
    """
    # 1) Try internal model_info helper (varies by Ultralytics version)
    try:
        from ultralytics.utils.torch_utils import model_info  # type: ignore
        info = model_info(yolo_model.model, verbose=False, imgsz=imgsz)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            flops = info[3]
            if flops is not None:
                flops = float(flops)
                return flops / 1e9 if flops > 1e6 else flops
    except Exception:
        pass

    # 2) Fallback: Ultralytics often logs via LOGGER (not stdout). Capture logs and parse '... GFLOPs'.
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
        raise RuntimeError(
            "Could not automatically extract GFLOPs from Ultralytics output. "
            "Try printing `model.info(imgsz=640)` and checking how it is logged in your environment."
        )
    return float(m.group(1))


def measure_forward_time_and_mem(yolo_model, imgsz: int = 640, bs: int = 1, warmup_iters: int = 10, measure_iters: int = 100):
    """Measure forward-only time (ms/img) and peak GPU memory (MB) using the underlying torch module."""
    # Ensure model and inputs are on the same device/dtype
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    yolo_model.model.to(device)
    yolo_model.model.eval()

    # Match input dtype to model weights dtype (e.g., FP16 vs FP32)
    w_dtype = next(yolo_model.model.parameters()).dtype
    x = torch.rand(bs, 3, imgsz, imgsz, device=device, dtype=w_dtype)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = yolo_model.model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Measure
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = yolo_model.model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_s = t1 - t0
    sec_per_img = total_s / (measure_iters * bs)
    ms_per_img = sec_per_img * 1e3

    peak_mem_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = yolo_model.model(x)
            torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    return ms_per_img, sec_per_img, peak_mem_mb


def report_ultralytics_metrics(yolo_model, weights_path: str, title: str, imgsz: int = 640):
    """Print the same metrics used in yolo/standard/test_yolo.py."""
    print("=" * 60)
    print(title)
    print("=" * 60)

    # Params
    n_params = sum(p.numel() for p in yolo_model.model.parameters())
    print(f"Params: {n_params} => {n_params / 1e6:.6f} M")

    # GFLOPs + analytical energy (paper-style)
    gflops = get_gflops_ultralytics(yolo_model, imgsz=imgsz)
    print(f"GFLOPs (imgsz={imgsz}): {gflops:.4f}")

    # Convention: 1 MAC = 2 FLOPs (common)
    E_MAC = 4.6e-12  # J (paper, 45nm, FP32)
    macs = gflops * 1e9 / 2
    E_j = macs * E_MAC
    print(f"MACs (G): {macs / 1e9:.4f}")
    print(f"Analytical energy (J/img): {E_j:.6f} J  ({E_j * 1e3:.3f} mJ)")

    # Forward-only time + Joules/img from measured power
    ms_per_img, sec_per_img, peak_mem_mb = measure_forward_time_and_mem(yolo_model, imgsz=imgsz)
    print(f"Infer time (forward-only, imgsz={imgsz}, bs=1): {ms_per_img:.3f} ms/img")

    # power_w = float(os.getenv("POWER_W", "0"))
    power_w = 6.05
    if power_w > 0:
        joules_per_img = power_w * sec_per_img
        print(f"Joules/img (POWER_W={power_w}): {joules_per_img:.6f} J  ({joules_per_img * 1e3:.3f} mJ)")
    else:
        print("Joules/img: set your measured power in Watts via env var POWER_W (e.g., export POWER_W=6.05)")

    if peak_mem_mb is not None:
        print(f"Peak GPU memory (forward-only): {peak_mem_mb:.1f} MB")

    # Checkpoint size on disk
    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"Checkpoint size: {size_mb:.1f} MB")

    # Optional: mAP (if DATA_YAML is provided)
    data_yaml = os.getenv("DATA_YAML", "").strip()
    if data_yaml:
        try:
            results = yolo_model.val(data=data_yaml, imgsz=imgsz)
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")
        except Exception as e:
            print(f"[WARN] Could not run validation (DATA_YAML={data_yaml}): {e}")

    print("=" * 60)
    print()

# Calcula tamanho do modelo original em memória
memory_size_orig = metrics.get_model_memory_size(model_nn)
print(f"Tamanho em memória (RAM): {memory_size_orig:.2f} MB")

# Conta parâmetros originais
total_params_orig, nonzero_params_orig = metrics.count_parameters(model_nn)
print(f"Total de parâmetros: {total_params_orig:,}")
print(f"Parâmetros não-zero: {nonzero_params_orig:,}")
print(f"Densidade: {(nonzero_params_orig/total_params_orig)*100:.2f}%")
print()

# Same metrics as yolo/standard/test_yolo.py (original model)
report_ultralytics_metrics(model, oxford_town_base_model, title="METRICS (ORIGINAL MODEL)", imgsz=640)

# -----------------------------
# CHANNEL PRUNING (real structured pruning)
# -----------------------------
# This performs true channel removal (changes tensor shapes), which can reduce Params, GFLOPs,
# inference time, and checkpoint size.

conv_layers = [(name, module) for name, module in model_nn.named_modules() if isinstance(module, nn.Conv2d)]

print("=" * 60)
print("APLICANDO PRUNING POR CANAL (CHANNEL PRUNING)")
print("=" * 60)
print(f"Total de camadas Conv2d encontradas: {len(conv_layers)}")

# Skip early/late layers (often hurts accuracy a lot)
skip_first = 2
skip_last = 2
prune_ratio = 0.30
min_channels = 8

# Device for dependency graph build
if torch.cuda.is_available():
    dg_device = torch.device("cuda:0")
else:
    dg_device = torch.device("cpu")

model_nn.to(dg_device)
model_nn.eval()

# Example input for dependency graph (must match model input)
example_inputs = torch.randn(1, 3, 640, 640, device=dg_device)

# Build dependency graph
DG = tp.DependencyGraph().build_dependency(model_nn, example_inputs=example_inputs)

pruned_any = 0
for idx, (name, conv) in enumerate(conv_layers):
    if idx < skip_first or idx >= len(conv_layers) - skip_last:
        print(f"Pulando camada {name}")
        continue

    # Heuristics: skip detection-specific heads / DFL / very small layers
    lname = name.lower()
    if "dfl" in lname or "detect" in lname:
        print(f"Pulando camada (head/dfl) {name}")
        continue

    out_ch = conv.out_channels
    if out_ch <= min_channels:
        print(f"Pulando camada (poucos canais: {out_ch}) {name}")
        continue

    n_prune = int(out_ch * prune_ratio)
    # keep at least min_channels
    if out_ch - n_prune < min_channels:
        n_prune = max(0, out_ch - min_channels)
    if n_prune <= 0:
        print(f"Pulando camada (n_prune=0) {name}")
        continue

    # Importance: L2 norm per output channel (filter)
    with torch.no_grad():
        w = conv.weight.detach()
        imp = torch.norm(w.reshape(out_ch, -1), p=2, dim=1)
        prune_idxs = torch.argsort(imp)[:n_prune].tolist()

    print(f"Pruning {n_prune}/{out_ch} canais em {name}")

    try:
        # torch-pruning API changed across versions.
        # Newer versions use `get_pruning_group(...).prune()`, older ones may have `get_pruning_plan(...).exec()`.
        if hasattr(DG, "get_pruning_plan"):
            plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channels, idxs=prune_idxs)
            plan.exec()
        elif hasattr(DG, "get_pruning_group"):
            group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=prune_idxs)
            group.prune()
        else:
            raise AttributeError("DependencyGraph has neither get_pruning_plan nor get_pruning_group")

        pruned_any += 1
    except Exception as e:
        print(f"[WARN] Falhou pruning em {name}: {e}")

print(f"Pruning executado em {pruned_any} camadas Conv2d")

# Sanity check: print a few Conv2d out_channels after pruning
try:
    sample_convs = [(n, m.out_channels) for n, m in model_nn.named_modules() if isinstance(m, nn.Conv2d)][:5]
    print("[Sanity] First Conv2d out_channels after pruning:", sample_convs)
except Exception as e:
    print(f"[Sanity][WARN] Could not list conv channels: {e}")

# Move back to CPU to free GPU memory before later measurements/models
model_nn.to("cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Verifica tamanho após pruning (antes de salvar)
print("=" * 60)
print("ESTATÍSTICAS APÓS PRUNING POR CANAL (EM MEMÓRIA)")
print("=" * 60)

# Calcula tamanho em memória após pruning
memory_size_pruned = metrics.get_model_memory_size(model_nn)
print(f"Tamanho em memória (RAM): {memory_size_pruned:.2f} MB")
print(f"Tamanho original em memória: {memory_size_orig:.2f} MB")
print(f"Diferença em memória: {memory_size_orig - memory_size_pruned:.2f} MB ({((memory_size_orig - memory_size_pruned)/memory_size_orig)*100:.2f}%)")
print()

# After channel pruning, parameters are physically removed (no sparsity mask)
total_params_pruned, nonzero_params_pruned = metrics.count_parameters(model_nn)
print(f"Total de parâmetros: {total_params_pruned:,}")
print(f"Parâmetros removidos: {total_params_orig - total_params_pruned:,}")
print(f"Redução de parâmetros: {((total_params_orig - total_params_pruned)/total_params_orig)*100:.2f}%")
print()

output_path = 'test_structured.pt'
model.save(output_path)

# Move original model off GPU before loading pruned model (fairer peak memory)
try:
    model.model.to("cpu")
except Exception:
    pass
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Reload pruned weights as a YOLO model and report the same metrics
pruned_model = YOLO(output_path)
# Fuse pruned model too (fair comparison)
try:
    pruned_model.fuse()
except Exception:
    try:
        pruned_model.model.fuse()
    except Exception:
        pass

report_ultralytics_metrics(pruned_model, output_path, title="METRICS (PRUNED MODEL)", imgsz=640)

print("=" * 60)
print("TAMANHO DO MODELO SALVO EM DISCO")
print("=" * 60)
final_size = metrics.get_model_size(output_path)
print(f"Tamanho do arquivo: {final_size:.2f} MB")
print(f"Tamanho original em disco: {original_size:.2f} MB")
print(f"Diferença em disco: {original_size - final_size:.2f} MB ({((original_size - final_size)/original_size)*100:.2f}%)")
print()

print(f"Modelo salvo com pruning estrutural aplicado em '{output_path}'")
print("=" * 60)