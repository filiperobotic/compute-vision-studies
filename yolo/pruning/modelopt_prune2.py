"""
Pruning YOLOv11 com NVIDIA Model Optimizer

Baseado em: https://y-t-g.github.io/tutorials/yolo-prune/

Requisitos:
    pip install nvidia-modelopt[torch]
    pip install git+https://github.com/ultralytics/ultralytics@qat-nvidia
"""

import torch
from ultralytics import YOLO
import modelopt.torch.prune as mtp
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA
import math


class PrunedTrainer:
    """
    Trainer customizado que aplica pruning com NVIDIA ModelOpt antes do treinamento
    """
    
    def __init__(self, base_trainer_class):
        """
        Args:
            base_trainer_class: Classe do trainer base do YOLO
        """
        self.base_trainer_class = base_trainer_class
    
    def __call__(self, *args, **kwargs):
        """
        Cria uma nova classe que herda do trainer base
        """
        base_trainer = self.base_trainer_class
        
        class CustomPrunedTrainer(base_trainer):
            def _setup_train(self):
                """
                Setup modificado que aplica pruning antes do treinamento
                """
                # Chama setup original
                super()._setup_train()
                
                LOGGER.info("="*60)
                LOGGER.info("Iniciando processo de pruning com NVIDIA ModelOpt...")
                LOGGER.info("="*60)
                
                # Função para coletar batches
                def collect_func(batch):
                    return self.preprocess_batch(batch)["img"]
                
                # Função de score (fitness) para avaliar subnets
                def score_func(model):
                    model.eval()
                    # Desabilita salvamento durante avaliação
                    save_orig = self.validator.args.save
                    plots_orig = self.validator.args.plots
                    verbose_orig = self.validator.args.verbose
                    
                    self.validator.args.save = False
                    self.validator.args.plots = False
                    self.validator.args.verbose = False
                    self.validator.is_coco = False
                    
                    metrics = self.validator(model=model)
                    
                    # Restaura configurações
                    self.validator.args.save = save_orig
                    self.validator.args.plots = plots_orig
                    self.validator.args.verbose = verbose_orig
                    
                    return metrics.fitness
                
                # Configurações de pruning
                # Você pode ajustar para "30%", "50%", "66%", etc.
                prune_constraints = {"flops": "66%"}
                
                LOGGER.info(f"Target de pruning: {prune_constraints}")
                
                # Desabilita fusing (necessário para subnet search)
                self.model.is_fused = lambda: True
                
                # Cria dummy input
                dummy_input = torch.randn(
                    1, 3, self.args.imgsz, self.args.imgsz
                ).to(self.device)
                
                # Aplica pruning
                LOGGER.info("Executando subnet search (pode demorar)...")
                
                self.model, prune_results = mtp.prune(
                    model=self.model,
                    mode="fastnas",
                    constraints=prune_constraints,
                    dummy_input=dummy_input,
                    config={
                        "score_func": score_func,
                        "checkpoint": "modelopt_fastnas_search_checkpoint.pth",
                        "data_loader": self.train_loader,
                        "collect_func": collect_func,
                        "max_iter_data_loader": 20,  # Use 50 para melhores resultados (requer mais RAM)
                    },
                )
                
                LOGGER.info("="*60)
                LOGGER.info("Pruning aplicado com sucesso!")
                LOGGER.info("="*60)
                
                # Move modelo para device
                self.model.to(self.device)
                
                # Recria EMA
                self.ema = ModelEMA(self.model)
                
                # Recria optimizer e scheduler
                weight_decay = (
                    self.args.weight_decay * 
                    self.batch_size * 
                    self.accumulate / 
                    self.args.nbs
                )
                
                iterations = (
                    math.ceil(
                        len(self.train_loader.dataset) / 
                        max(self.batch_size, self.args.nbs)
                    ) * self.epochs
                )
                
                self.optimizer = self.build_optimizer(
                    model=self.model,
                    name=self.args.optimizer,
                    lr=self.args.lr0,
                    momentum=self.args.momentum,
                    decay=weight_decay,
                    iterations=iterations,
                )
                
                self._setup_scheduler()
                
                LOGGER.info("Optimizer e scheduler recriados")
        
        return CustomPrunedTrainer(*args, **kwargs)


def train_with_pruning(
    model_path="yolo11n.pt",
    data_yaml="coco128.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    flops_target="66%",
    project="runs/prune",
    name="pruned_model"
):
    """
    Treina modelo YOLO com pruning usando NVIDIA ModelOpt
    
    Args:
        model_path: Caminho para modelo YOLO
        data_yaml: Dataset YAML
        epochs: Número de épocas
        imgsz: Tamanho da imagem
        batch: Batch size
        flops_target: Target de FLOPs (ex: "30%", "50%", "66%")
        project: Diretório do projeto
        name: Nome do experimento
    
    Returns:
        Resultados do treinamento
    """
    print("\n" + "="*70)
    print("YOLO PRUNING COM NVIDIA MODEL OPTIMIZER")
    print("="*70)
    
    # Carrega modelo
    print(f"\nCarregando modelo: {model_path}")
    model = YOLO(model_path)
    
    # Info do modelo original
    print("\nMODELO ORIGINAL:")
    model.info()
    
    # Obtém classe do trainer
    trainer_class = model.task_map[model.task]["trainer"]
    
    # Cria trainer com pruning
    pruned_trainer = PrunedTrainer(trainer_class)
    
    # Treina com pruning
    print(f"\n{'='*70}")
    print(f"INICIANDO TREINAMENTO COM PRUNING")
    print(f"Target: {flops_target} FLOPs")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        trainer=pruned_trainer,
        project=project,
        name=name,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("TREINAMENTO COMPLETO!")
    print("="*70)
    
    return results


def evaluate_pruned_model(model_path, data_yaml="coco128.yaml"):
    """
    Avalia modelo pruned
    
    Args:
        model_path: Caminho para modelo pruned
        data_yaml: Dataset YAML
    """
    print("\n" + "="*70)
    print("AVALIANDO MODELO PRUNED")
    print("="*70)
    
    model = YOLO(model_path)
    
    print("\nINFO DO MODELO:")
    model.info()
    
    print("\nMÉTRICAS DE VALIDAÇÃO:")
    metrics = model.val(data=data_yaml)
    
    return metrics


def compare_models(original_path, pruned_path, data_yaml="coco128.yaml"):
    """
    Compara modelo original vs pruned
    
    Args:
        original_path: Caminho modelo original
        pruned_path: Caminho modelo pruned
        data_yaml: Dataset YAML
    """
    print("\n" + "="*70)
    print("COMPARAÇÃO: ORIGINAL vs PRUNED")
    print("="*70)
    
    # Carrega modelos
    original = YOLO(original_path)
    pruned = YOLO(pruned_path)
    
    print("\n📊 MODELO ORIGINAL:")
    print("-" * 70)
    original.info()
    
    print("\n📊 MODELO PRUNED:")
    print("-" * 70)
    pruned.info()
    
    # Métricas
    print("\n📈 AVALIANDO PERFORMANCE...")
    print("-" * 70)
    
    print("\nOriginal:")
    orig_metrics = original.val(data=data_yaml, verbose=False)
    
    print("\nPruned:")
    pruned_metrics = pruned.val(data=data_yaml, verbose=False)
    
    # Resumo
    print("\n" + "="*70)
    print("📊 RESUMO DA COMPARAÇÃO")
    print("="*70)
    
    print("\nmAP50:")
    print(f"  Original: {orig_metrics.box.map50:.4f}")
    print(f"  Pruned:   {pruned_metrics.box.map50:.4f}")
    print(f"  Diferença: {pruned_metrics.box.map50 - orig_metrics.box.map50:+.4f}")
    
    print("\nmAP50-95:")
    print(f"  Original: {orig_metrics.box.map:.4f}")
    print(f"  Pruned:   {pruned_metrics.box.map:.4f}")
    print(f"  Diferença: {pruned_metrics.box.map - orig_metrics.box.map:+.4f}")
    
    # Tamanho dos arquivos
    import os
    orig_size = os.path.getsize(original_path) / (1024**2)
    pruned_size = os.path.getsize(pruned_path) / (1024**2)
    
    print("\nTamanho do arquivo:")
    print(f"  Original: {orig_size:.2f} MB")
    print(f"  Pruned:   {pruned_size:.2f} MB")
    print(f"  Redução:  {(1 - pruned_size/orig_size)*100:.2f}%")
    
    print("\n" + "="*70)


# Exemplo de uso
if __name__ == "__main__":
    # Configurações
    MODEL_PATH = "yolo11x.pt"
    DATA_YAML = "data.yaml"  # Use seu dataset aqui
    EPOCHS = 50  # Aumente para seu dataset real
    BATCH_SIZE = 16
    FLOPS_TARGET = "66%"  # sPode testar 30%, 50%, 66%
    
    # 1. Treina com pruning
    print("\n" + "="*70)
    print("ETAPA 1: TREINAMENTO COM PRUNING")
    print("="*70)
    
    results = train_with_pruning(
        model_path=MODEL_PATH,
        data_yaml=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        flops_target=FLOPS_TARGET,
        project="runs/prune",
        name="yolo11x_pruned"
    )
    
    # 2. Avalia modelo pruned
    print("\n" + "="*70)
    print("ETAPA 2: AVALIAÇÃO DO MODELO PRUNED")
    print("="*70)
    
    pruned_model_path = "runs/prune/yolo11x_pruned/weights/best.pt"
    metrics = evaluate_pruned_model(pruned_model_path, DATA_YAML)
    
    # 3. Compara com original
    print("\n" + "="*70)
    print("ETAPA 3: COMPARAÇÃO DETALHADA")
    print("="*70)
    
    compare_models(MODEL_PATH, pruned_model_path, DATA_YAML)
    
    # 4. Instruções finais
    print("\n" + "="*70)
    print("✅ PROCESSO COMPLETO!")
    print("="*70)
    
    print("\n💡 PRÓXIMOS PASSOS:")
    print("  1. Exporte para TensorRT FP16 para ganhos de velocidade:")
    print("     pruned_model.export(format='engine', half=True)")
    print("")
    print("  2. Para carregar o modelo pruned posteriormente:")
    print("     from ultralytics import YOLO")
    print(f"     model = YOLO('{pruned_model_path}')")
    print("")
    print("  3. Ajuste FLOPS_TARGET para diferentes níveis de compressão:")
    print("     - 30%: Mais agressivo (menor modelo, maior perda)")
    print("     - 50%: Balanceado")
    print("     - 66%: Conservador (recomendado para começar)")
    print("")
    print("  4. IMPORTANTE: Use seu dataset completo para melhores resultados")
    print(f"     (este exemplo usa {DATA_YAML} apenas para demonstração)")
    
    print("\n" + "="*70)