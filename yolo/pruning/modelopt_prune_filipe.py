from ultralytics import YOLO
import torch, os, time, io, re, logging
from contextlib import redirect_stdout
from pathlib import Path
from collections import defaultdict
from utils import metrics

# --------- CONFIG ---------
WEIGHTS = os.getenv("WEIGHTS", "/home/pesquisador/pesquisa/filipe/compute-vision-studies/runs/train/yolo11x__oxford_tower_custom_train/weights/best.pt")
DATA_YAML = os.getenv("DATA_YAML", "data.yaml")
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))

FLOPS_TARGET = os.getenv("FLOPS_TARGET", "66%")          # guia usa "66%"  [oai_citation:3‡Yasin's Keep](https://y-t-g.github.io/tutorials/yolo-prune/)
MAX_ITER_DATALOADER = int(os.getenv("MAX_ITER_DATALOADER", "20"))  # guia cita 20 (50 recomendado, mas RAM)  [oai_citation:4‡Yasin's Keep](https://y-t-g.github.io/tutorials/yolo-prune/)
FINETUNE_EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "10"))
POWER_W = float(os.getenv("POWER_W", "6.05"))

# --------- METRICS (copiadas do seu script atual) ---------
def get_gflops_ultralytics(yolo_model, imgsz: int = 640) -> float:
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

def measure_forward_time_and_mem(yolo_model, imgsz: int = 640, bs: int = 1, warmup_iters: int = 10, measure_iters: int = 100):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    yolo_model.model.to(device)
    yolo_model.model.eval()

    w_dtype = next(yolo_model.model.parameters()).dtype
    x = torch.rand(bs, 3, imgsz, imgsz, device=device, dtype=w_dtype)

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
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = yolo_model.model(x)
            torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    return ms_per_img, sec_per_img, peak_mem_mb

def report_ultralytics_metrics(yolo_model, weights_path: str, title: str, imgsz: int = 640):
    print("=" * 60)
    print(title)
    print("=" * 60)

    n_params = sum(p.numel() for p in yolo_model.model.parameters())
    print(f"Params: {n_params} => {n_params / 1e6:.6f} M")

    gflops = get_gflops_ultralytics(yolo_model, imgsz=imgsz)
    print(f"GFLOPs (imgsz={imgsz}): {gflops:.4f}")

    E_MAC = 4.6e-12  # J
    macs = gflops * 1e9 / 2
    E_j = macs * E_MAC
    print(f"MACs (G): {macs / 1e9:.4f}")
    print(f"Analytical energy (J/img): {E_j:.6f} J  ({E_j * 1e3:.3f} mJ)")

    ms_per_img, sec_per_img, peak_mem_mb = measure_forward_time_and_mem(yolo_model, imgsz=imgsz)
    print(f"Infer time (forward-only, imgsz={imgsz}, bs=1): {ms_per_img:.3f} ms/img")

    if POWER_W > 0:
        joules_per_img = POWER_W * sec_per_img
        print(f"Joules/img (POWER_W={POWER_W}): {joules_per_img:.6f} J  ({joules_per_img * 1e3:.3f} mJ)")

    if peak_mem_mb is not None:
        print(f"Peak GPU memory (forward-only): {peak_mem_mb:.1f} MB")

    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"Checkpoint size: {size_mb:.1f} MB")

    print("=" * 60)
    print()

# --------- MODELOPT PRUNING (como no guia) ---------
def main():
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"weights not found: {WEIGHTS}")

    model = YOLO(WEIGHTS)
    try:
        model.fuse()
    except Exception:
        pass

    # baseline stats
    orig_nn = model.model
    memory_size_orig = metrics.get_model_memory_size(orig_nn)
    total_params_orig, _ = metrics.count_parameters(orig_nn)

    print("=" * 60)
    print("ORIGINAL")
    print("=" * 60)
    print(f"Checkpoint size: {metrics.get_model_size(WEIGHTS):.2f} MB")
    print(f"RAM size (rough): {memory_size_orig:.2f} MB")
    print(f"Params: {total_params_orig:,}")
    report_ultralytics_metrics(model, WEIGHTS, "METRICS (ORIGINAL MODEL)", imgsz=IMG_SIZE)

    # Ultralytics ModelOpt/QAT integration may require newer PyTorch.
    # Your traceback showed: AssertionError: QAT requires PyTorch>=2.6
    v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if v < (2, 6):
        print("[WARN] Your torch version is", torch.__version__)
        print("       Ultralytics ModelOpt/QAT integration may assert: 'QAT requires PyTorch>=2.6'.")
        print("       Quick fix: upgrade torch/torchvision to >=2.6 (matching your CUDA), OR run training with val=False.")
    # PyTorch 2.6 changed torch.load default to weights_only=True, which can break ModelOpt's search checkpoint
    # unpickling (e.g., collections.defaultdict). If the checkpoint is local/trusted, allowlist defaultdict.
    try:
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([defaultdict])
    except Exception as e:
        print(f"[WARN] Could not add_safe_globals(defaultdict): {e}")

    import modelopt.torch.prune as mtp  # needs nvidia-modelopt
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import ModelEMA
    import math

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
                self.validator.is_coco = False
                out = self.validator(model=m)
                self.validator.args.save = self.args.save
                self.validator.args.plots = self.args.plots
                self.validator.args.verbose = self.args.verbose
                return out["fitness"]

            prune_constraints = {"flops": FLOPS_TARGET}  #  [oai_citation:5‡Yasin's Keep](https://y-t-g.github.io/tutorials/yolo-prune/)
            self.model.is_fused = lambda: True  # disable fusing  [oai_citation:6‡Yasin's Keep](https://y-t-g.github.io/tutorials/yolo-prune/)

            ckpt_path = f"modelopt_fastnas_search_checkpoint_torch{torch.__version__.split('+')[0]}.pth"
            if os.path.exists(ckpt_path):
                print(f"[ModelOpt] Removing existing search checkpoint (may be incompatible): {ckpt_path}")
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"[WARN] Could not remove {ckpt_path}: {e}")

            self.model, _ = mtp.prune(
                model=self.model,
                mode="fastnas",
                constraints=prune_constraints,
                dummy_input=torch.randn(1, 3, self.args.imgsz, self.args.imgsz).to(self.device),
                config={
                    "score_func": score_func,
                    "checkpoint": f"modelopt_fastnas_search_checkpoint_torch{torch.__version__.split('+')[0]}.pth",
                    "data_loader": self.train_loader,
                    "collect_func": collect_func,
                    "max_iter_data_loader": MAX_ITER_DATALOADER,
                },
            )

            self.model.to(self.device)
            self.ema = ModelEMA(self.model)

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
            LOGGER.info(f"[ModelOpt] Applied pruning constraints: {prune_constraints}")

    # finetune pruned model
    # NOTE: Ultralytics runs a final_eval() that can trigger ModelOpt/QAT checks when modelopt is installed.
    # Setting val=False avoids the final validation AutoBackend load path that raised 'QAT requires PyTorch>=2.6'.
    train_res = model.train(
        data=DATA_YAML,
        trainer=PrunedTrainer,
        epochs=FINETUNE_EPOCHS,
        imgsz=IMG_SIZE,
        val=False,
    )
    print("[INFO] Training completed with val=False (skipped Ultralytics final_eval).")

    # find best.pt
    save_dir = getattr(train_res, "save_dir", None)
    if save_dir is None and isinstance(train_res, dict):
        save_dir = train_res.get("save_dir", None)

    pruned_best = None
    if save_dir is not None:
        cand = Path(str(save_dir)) / "weights" / "best.pt"
        if cand.exists():
            pruned_best = str(cand)

    if pruned_best is None:
        candidates = sorted(Path("runs").rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        pruned_best = str(candidates[0]) if candidates else None

    if pruned_best is None or not os.path.exists(pruned_best):
        raise RuntimeError("Could not find pruned best.pt after training.")

    pruned_model = YOLO(pruned_best)
    try:
        pruned_model.fuse()
    except Exception:
        pass

    pruned_nn = pruned_model.model
    memory_size_pruned = metrics.get_model_memory_size(pruned_nn)
    total_params_pruned, _ = metrics.count_parameters(pruned_nn)

    print("=" * 60)
    print("PRUNED (MODELOPT)")
    print("=" * 60)
    print(f"Pruned weights: {pruned_best}")
    print(f"Checkpoint size: {metrics.get_model_size(pruned_best):.2f} MB")
    print(f"RAM size (rough): {memory_size_pruned:.2f} MB")
    print(f"Params: {total_params_pruned:,}")
    print(f"Params removed: {total_params_orig - total_params_pruned:,}")
    report_ultralytics_metrics(pruned_model, pruned_best, "METRICS (PRUNED MODEL)", imgsz=IMG_SIZE)

if __name__ == "__main__":
    main()