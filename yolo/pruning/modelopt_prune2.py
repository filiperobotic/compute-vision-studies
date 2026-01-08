import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from pathlib import Path

class YOLOPruner:
    """
    Classe para realizar pruning estruturado em modelos YOLOv11
    """
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Inicializa o pruner
        
        Args:
            model_path: Caminho para o modelo YOLOv11
            device: Dispositivo para executar (cuda/cpu)
        """
        self.device = device
        self.model = YOLO(model_path)
        self.original_model = self.model.model.to(device)
        
    def calculate_channel_importance(self, module):
        """
        Calcula a importância de cada canal baseado na norma L1 dos pesos
        
        Args:
            module: Módulo de convolução
            
        Returns:
            Array com importância de cada canal
        """
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data.cpu().numpy()
            # Calcula norma L1 para cada filtro de saída
            importance = np.sum(np.abs(weight), axis=(1, 2, 3))
            return importance
        return None
    
    def prune_conv_layer(self, layer, prune_ratio):
        """
        Realiza pruning em uma camada convolucional
        
        Args:
            layer: Camada convolucional
            prune_ratio: Percentual de canais a remover (0-1)
            
        Returns:
            Máscara dos canais mantidos
        """
        if not isinstance(layer, nn.Conv2d):
            return None
            
        importance = self.calculate_channel_importance(layer)
        num_channels = len(importance)
        num_prune = int(num_channels * prune_ratio)
        
        if num_prune == 0:
            return torch.ones(num_channels, dtype=torch.bool)
        
        # Identifica os canais menos importantes
        threshold = np.sort(importance)[num_prune]
        mask = torch.from_numpy(importance > threshold).to(self.device)
        
        return mask
    
    def apply_pruning(self, prune_ratio=0.3, layer_types=['Conv']):
        """
        Aplica pruning ao modelo
        
        Args:
            prune_ratio: Percentual de canais a remover (0-1)
            layer_types: Tipos de camadas a aplicar pruning
            
        Returns:
            Modelo com pruning aplicado
        """
        print(f"Iniciando pruning com ratio: {prune_ratio}")
        pruned_modules = 0
        
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Conv2d) and any(lt in name for lt in layer_types):
                mask = self.prune_conv_layer(module, prune_ratio)
                
                if mask is not None and mask.sum() < len(mask):
                    # Aplica máscara aos pesos
                    module.weight.data *= mask.view(-1, 1, 1, 1).float()
                    if module.bias is not None:
                        module.bias.data *= mask.float()
                    
                    pruned_modules += 1
                    kept_channels = mask.sum().item()
                    total_channels = len(mask)
                    print(f"  {name}: {kept_channels}/{total_channels} canais mantidos")
        
        print(f"\nTotal de módulos com pruning: {pruned_modules}")
        return self.model
    
    def fine_tune(self, data_yaml, epochs=10, imgsz=640, batch=16):
        """
        Fine-tuning do modelo após pruning
        
        Args:
            data_yaml: Caminho para arquivo YAML do dataset
            epochs: Número de épocas
            imgsz: Tamanho da imagem
            batch: Tamanho do batch
        """
        print(f"\nIniciando fine-tuning por {epochs} épocas...")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            patience=50,
            save=True,
            verbose=True
        )
        return results
    
    def evaluate(self, data_yaml, imgsz=640):
        """
        Avalia o modelo
        
        Args:
            data_yaml: Caminho para arquivo YAML do dataset
            imgsz: Tamanho da imagem
            
        Returns:
            Métricas de avaliação
        """
        print("\nAvaliando modelo...")
        metrics = self.model.val(
            data=data_yaml,
            imgsz=imgsz,
            device=self.device
        )
        return metrics
    
    def get_model_size(self):
        """
        Calcula o tamanho do modelo em MB
        
        Returns:
            Tamanho em MB
        """
        param_size = 0
        for param in self.original_model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.original_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def count_parameters(self):
        """
        Conta o número de parâmetros do modelo
        
        Returns:
            Número total de parâmetros
        """
        return sum(p.numel() for p in self.original_model.parameters())
    
    def save_pruned_model(self, output_path):
        """
        Salva o modelo com pruning
        
        Args:
            output_path: Caminho para salvar o modelo
        """
        self.model.save(output_path)
        print(f"\nModelo salvo em: {output_path}")


# Exemplo de uso
if __name__ == "__main__":
    # Configurações
    MODEL_PATH = "yolo11x.pt"  # Caminho do modelo original
    DATA_YAML = "data.yaml"     # Arquivo YAML do dataset
    PRUNE_RATIO = 0.3           # 30% dos canais serão removidos
    OUTPUT_PATH = "yolo11x_pruned.pt"

    # Criar pruner
    pruner = YOLOPruner(MODEL_PATH)
    
    # Informações do modelo original
    print("=" * 50)
    print("MODELO ORIGINAL")
    print("=" * 50)
    print(f"Parâmetros: {pruner.count_parameters():,}")
    print(f"Tamanho: {pruner.get_model_size():.2f} MB")
    
    # Avaliar modelo original
    print("\nAvaliando modelo original...")
    original_metrics = pruner.evaluate(DATA_YAML)
    
    # Aplicar pruning
    print("\n" + "=" * 50)
    print("APLICANDO PRUNING")
    print("=" * 50)
    pruned_model = pruner.apply_pruning(prune_ratio=PRUNE_RATIO)
    
    # Informações do modelo com pruning
    print("\n" + "=" * 50)
    print("MODELO COM PRUNING")
    print("=" * 50)
    print(f"Parâmetros: {pruner.count_parameters():,}")
    print(f"Tamanho: {pruner.get_model_size():.2f} MB")
    
    # Fine-tuning
    print("\n" + "=" * 50)
    print("FINE-TUNING")
    print("=" * 50)
    pruner.fine_tune(DATA_YAML, epochs=10, batch=16)
    
    # Avaliar modelo após fine-tuning
    print("\n" + "=" * 50)
    print("AVALIAÇÃO FINAL")
    print("=" * 50)
    final_metrics = pruner.evaluate(DATA_YAML)
    
    # Salvar modelo
    pruner.save_pruned_model(OUTPUT_PATH)
    
    # Resumo
    print("\n" + "=" * 50)
    print("RESUMO")
    print("=" * 50)
    print(f"Redução de parâmetros: {(1 - pruner.count_parameters()/pruner.count_parameters())*100:.2f}%")
    print(f"mAP50 Original: {original_metrics.box.map50:.4f}")
    print(f"mAP50 Final: {final_metrics.box.map50:.4f}")
    print(f"Diferença: {(final_metrics.box.map50 - original_metrics.box.map50):.4f}")