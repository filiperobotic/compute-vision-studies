import os
from collections import defaultdict
import torch

# PyTorch 2.6 changed torch.load default to weights_only=True.
# ModelOpt FastNAS search checkpoints may contain pickled objects like collections.defaultdict.
# Allowlist it so weights_only unpickling succeeds.
try:
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([defaultdict])
        print("[INFO] Added safe global: collections.defaultdict")
except Exception as e:
    print(f"[WARN] Could not add_safe_globals([defaultdict]): {e}")

from ultralytics import YOLO
import modelopt.torch.prune as mtp

model = YOLO("yolo11x.pt")

class PrunedTrainer(model.task_map[model.task]["trainer"]):
    def _setup_train(self):
        """Modified setup model that adds distillation wrapper."""
        from ultralytics.utils import LOGGER
        from ultralytics.utils.torch_utils import ModelEMA
        from ultralytics import YOLO
        from collections import OrderedDict
        import torch
        import math

        super()._setup_train()

        def collect_func(batch):
            return self.preprocess_batch(batch)["img"]

        def score_func(model):
            # Disable logs
            model.eval()
            self.validator.args.save = False
            self.validator.args.plots = False
            self.validator.args.verbose = False
            self.validator.args.data = "coco128.yaml"
            metrics = self.validator(model=model)
            self.validator.args.save = self.args.save
            self.validator.args.plots = self.args.plots
            self.validator.args.verbose = self.args.verbose
            self.validator.args.data = self.args.data
            return metrics["fitness"]

        prune_constraints = {"flops": "66%"}  # prune to 66% of original FLOPs

        ckpt_path = "modelopt_fastnas_search_checkpoint.pth"
        if os.path.exists(ckpt_path):
            print(f"[ModelOpt] Removing existing search checkpoint: {ckpt_path}")
            try:
                os.remove(ckpt_path)
            except Exception as e:
                print(f"[WARN] Could not remove {ckpt_path}: {e}")

        self.model.is_fused = lambda: True  # disable fusing

        self.model, prune_res = mtp.prune(
            model=self.model,
            mode="fastnas",
            constraints=prune_constraints,
            dummy_input=torch.randn(1, 3, self.args.imgsz, self.args.imgsz).to(self.device),
            config={
                "score_func": score_func,  # scoring function
                "checkpoint": ckpt_path,  # saves checkpoint during subnet search
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

    def final_eval(self):
        # Disable Ultralytics final_eval for ModelOpt-pruned runs.
        # final_eval reloads best.pt via AutoBackend, which triggers ModelOpt restore and can crash
        # with "Inconsistent keys in config".
        return


# Treinar o modelo
results = model.train(data="./data.yaml", trainer=PrunedTrainer, epochs=50, exist_ok=True, warmup_epochs=0, val=False)

# ============================================================================
# SOLUÇÃO: Exportar modelo podado para uso no Raspberry Pi
# ============================================================================

print("\n[INFO] Finalizando arquitetura do modelo podado...")
import modelopt.torch.opt as mto

# Usar o modelo que acabou de treinar
pruned_model = model

# Finalizar a arquitetura (remove a search space do ModelOpt)
mto.export_for_deployment(pruned_model.model)

print("\n[INFO] Salvando modelo podado limpo para Raspberry Pi...")

# Criar um checkpoint limpo compatível com YOLO padrão
import copy
from pathlib import Path

save_dir = Path("pruned_models")
save_dir.mkdir(exist_ok=True)

# Caminho do modelo limpo
clean_model_path = save_dir / "yolo11x_pruned_clean.pt"

# Criar um checkpoint no formato esperado pelo Ultralytics
checkpoint = {
    'epoch': 50,
    'model': pruned_model.model.state_dict(),
    'optimizer': None,
    'best_fitness': None,
    'ema': None,
    'updates': None,
    'date': None,
}

# Salvar o checkpoint limpo
torch.save(checkpoint, clean_model_path)
print(f"[INFO] Modelo limpo salvo em: {clean_model_path}")

# ============================================================================
# Validar que o modelo pode ser carregado sem erros
# ============================================================================
print("\n[INFO] Testando carregamento do modelo limpo...")
try:
    # Tentar carregar o modelo limpo
    test_model = YOLO(str(clean_model_path))
    print("[SUCCESS] Modelo carregado com sucesso!")
    
    # Mostrar informações
    print("\n[INFO] Informações do modelo ORIGINAL:")
    original_model = YOLO("yolo11x.pt")
    original_model.info()
    
    print("\n[INFO] Informações do modelo PODADO:")
    test_model.info()
    
    # Validação
    print("\n[INFO] Executando validação no modelo podado...")
    results = test_model.val(data="./data.yaml", verbose=False, rect=False)
    
    print(f"\n[INFO] mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"[INFO] mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    
except Exception as e:
    print(f"[ERROR] Falha ao carregar modelo: {e}")
    print("[INFO] Tentando método alternativo...")
    
    # Método alternativo: salvar apenas state_dict e recriar modelo base
    state_dict_path = save_dir / "yolo11x_pruned_state.pt"
    torch.save(pruned_model.model.state_dict(), state_dict_path)
    print(f"[INFO] State dict salvo em: {state_dict_path}")
    print("[INFO] Você precisará carregar este state_dict manualmente em um modelo YOLO")

# ============================================================================
# EXPORTAR para formatos otimizados para Raspberry Pi
# ============================================================================
print("\n[INFO] Exportando para formatos otimizados para Raspberry Pi...")

# Recarregar o modelo limpo
final_model = YOLO(str(clean_model_path))

# 1. ONNX (recomendado para Raspberry Pi com ONNX Runtime)
try:
    print("[INFO] Exportando para ONNX...")
    final_model.export(format="onnx", simplify=True, dynamic=False, imgsz=640)
    print("[SUCCESS] Modelo ONNX salvo!")
except Exception as e:
    print(f"[WARN] Erro ao exportar ONNX: {e}")

# 2. TorchScript (alternativa)
try:
    print("[INFO] Exportando para TorchScript...")
    final_model.export(format="torchscript", imgsz=640)
    print("[SUCCESS] Modelo TorchScript salvo!")
except Exception as e:
    print(f"[WARN] Erro ao exportar TorchScript: {e}")

# 3. TFLite (ideal para edge devices como Raspberry Pi)
try:
    print("[INFO] Exportando para TFLite...")
    final_model.export(format="tflite", imgsz=640, int8=False)
    print("[SUCCESS] Modelo TFLite salvo!")
except Exception as e:
    print(f"[WARN] Erro ao exportar TFLite: {e}")

print("\n" + "="*60)
print("RESUMO DOS MODELOS EXPORTADOS:")
print("="*60)
print(f"1. PyTorch (.pt):      {clean_model_path}")
print(f"2. ONNX (.onnx):       {clean_model_path.parent / clean_model_path.stem}.onnx")
print(f"3. TorchScript (.ts):  {clean_model_path.parent / clean_model_path.stem}.torchscript")
print(f"4. TFLite (.tflite):   {clean_model_path.parent / clean_model_path.stem}.tflite")
print("="*60)
print("\nPara Raspberry Pi, recomendo usar o modelo ONNX ou TFLite")
print("="*60)