from ultralytics import YOLO
import modelopt.torch.prune as mtp
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics import YOLO
from collections import OrderedDict, defaultdict
import torch
import math
import os

model = YOLO("yolov8m.pt")

# PyTorch 2.6 sets torch.load(weights_only=True) by default, which can break unpickling of
# ModelOpt search checkpoints that contain objects like collections.defaultdict.
# We explicitly allowlist defaultdict (safe if the checkpoint is local/trusted).
try:
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([defaultdict])
        print("[INFO] Added safe global: collections.defaultdict")
except Exception as e:
    print(f"[WARN] Could not add_safe_globals([defaultdict]): {e}")

class PrunedTrainer(model.task_map[model.task]["trainer"]):
  def _setup_train(self):
    """Modified setup model that adds distillation wrapper."""
    

    super()._setup_train()

    def collect_func(batch):
        return self.preprocess_batch(batch)["img"]

    def score_func(model):
        # Disable logs
        model.eval()
        self.validator.args.save = False
        self.validator.args.plots = False
        self.validator.args.verbose = False
        self.validator.args.data = "data.yaml"
        metrics = self.validator(model=model)
        self.validator.args.save = self.args.save
        self.validator.args.plots = self.args.plots
        self.validator.args.verbose = self.args.verbose
        self.validator.args.data = self.args.data
        return metrics["fitness"]

    # Use a versioned checkpoint name to avoid loading stale/incompatible checkpoints
    ckpt_name = f"modelopt_fastnas_search_checkpoint_torch{torch.__version__.split('+')[0]}.pth"
    if os.path.exists(ckpt_name):
        print(f"[ModelOpt] Removing existing search checkpoint (may be incompatible): {ckpt_name}")
        try:
            os.remove(ckpt_name)
        except Exception as e:
            print(f"[WARN] Could not remove {ckpt_name}: {e}")

    prune_constraints = {"flops": "66%"}  # prune to 66% of original FLOPs

    self.model.is_fused = lambda: True  # disable fusing

    self.model, prune_res = mtp.prune(
        model=self.model,
        mode="fastnas",
        constraints=prune_constraints,
        dummy_input=torch.randn(1, 3, self.args.imgsz, self.args.imgsz).to(self.device),
        config={
            "score_func": score_func,  # scoring function
            "checkpoint": ckpt_name,  # saves checkpoint during subnet search
            "data_loader": self.train_loader,  # training dataloader
            "collect_func": collect_func,  # preprocessing function
            "max_iter_data_loader": 20,  # 50 is recommended, but requires more RAM
        },
    )

    self.model.to(self.device)
    # Recreate EMA
    self.ema = ModelEMA(self.model)

    # Recreate optimizer and scheduler
    weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
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

results = model.train(data="data.yaml", trainer=PrunedTrainer, epochs=50, exist_ok=True, warmup_epochs=0)
pruned_model = YOLO("runs/detect/train/weights/best.pt")

# Original FLOPs
model = YOLO("yolov8m.pt")
model.info()

# Pruned FLOPs
pruned_model.info()

results = pruned_model.val(data="data.yaml", verbose=False, rect=False)

pruned_model.export(format="engine", half=True)
pruned_model = YOLO("runs/detect/train/weights/best.engine")
results = pruned_model.val(data="data.yaml", verbose=False)

model = YOLO("yolov8m.pt")
model.export(format="engine", half=True)
model = YOLO("yolov8m.engine")

results = model.val(data="data.yaml", verbose=False)