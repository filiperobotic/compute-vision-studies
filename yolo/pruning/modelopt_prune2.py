import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
import numpy as np
from copy import deepcopy

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
        self.pruning_masks = {}
        
    def calculate_channel_importance(self, conv_layer):
        """
        Calcula a importância de cada canal baseado na norma L1 dos pesos
        
        Args:
            conv_layer: Camada convolucional
            
        Returns:
            Array com importância de cada canal
        """
        weight = conv_layer.weight.data.cpu().numpy()
        # Calcula norma L1 para cada filtro de saída
        importance = np.sum(np.abs(weight), axis=(1, 2, 3))
        return importance
    
    def get_pruning_mask(self, conv_layer, prune_ratio):
        """
        Gera máscara de pruning para uma camada
        
        Args:
            conv_layer: Camada convolucional
            prune_ratio: Percentual de canais a remover (0-1)
            
        Returns:
            Máscara booleana dos canais a manter
        """
        importance = self.calculate_channel_importance(conv_layer)
        num_channels = len(importance)
        num_keep = max(1, int(num_channels * (1 - prune_ratio)))  # Mantém pelo menos 1 canal
        
        # Índices dos canais mais importantes
        keep_indices = np.argsort(importance)[-num_keep:]
        
        mask = np.zeros(num_channels, dtype=bool)
        mask[keep_indices] = True
        
        return mask
    
    def prune_conv_layer(self, conv_layer, mask_in, mask_out):
        """
        Cria nova camada Conv com canais reduzidos
        
        Args:
            conv_layer: Camada original
            mask_in: Máscara de entrada
            mask_out: Máscara de saída
            
        Returns:
            Nova camada Conv com pruning aplicado
        """
        # Extrai parâmetros da camada original
        in_channels = int(mask_in.sum()) if mask_in is not None else conv_layer.in_channels
        out_channels = int(mask_out.sum())
        
        # Cria nova camada
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=1,  # Simplifica grupos após pruning
            bias=conv_layer.bias is not None
        ).to(self.device)
        
        # Copia pesos filtrados
        old_weight = conv_layer.weight.data.cpu().numpy()
        
        if mask_in is not None:
            old_weight = old_weight[:, mask_in, :, :]
        
        new_weight = old_weight[mask_out, :, :, :]
        new_conv.weight.data = torch.from_numpy(new_weight).to(self.device)
        
        # Copia bias se existir
        if conv_layer.bias is not None:
            old_bias = conv_layer.bias.data.cpu().numpy()
            new_bias = old_bias[mask_out]
            new_conv.bias.data = torch.from_numpy(new_bias).to(self.device)
        
        return new_conv
    
    def analyze_model_structure(self):
        """
        Analisa a estrutura do modelo e identifica camadas para pruning
        
        Returns:
            Dicionário com informações das camadas
        """
        layer_info = {}
        
        for i, module in enumerate(self.original_model.model):
            module_type = type(module).__name__
            
            if isinstance(module, Conv):
                layer_info[i] = {
                    'type': 'Conv',
                    'conv': module.conv,
                    'out_channels': module.conv.out_channels
                }
            elif isinstance(module, C2f):
                layer_info[i] = {
                    'type': 'C2f',
                    'cv1': module.cv1.conv if hasattr(module.cv1, 'conv') else module.cv1,
                    'cv2': module.cv2.conv if hasattr(module.cv2, 'conv') else module.cv2,
                }
            
        return layer_info
    
    def compute_pruning_plan(self, prune_ratio=0.3, skip_layers=None):
        """
        Computa plano de pruning para todas as camadas
        
        Args:
            prune_ratio: Ratio de pruning
            skip_layers: Lista de índices de camadas a não fazer pruning
            
        Returns:
            Dicionário com máscaras de pruning
        """
        if skip_layers is None:
            skip_layers = []
        
        masks = {}
        layer_info = self.analyze_model_structure()
        
        print(f"\nComputando plano de pruning (ratio={prune_ratio})...")
        print("=" * 60)
        
        for idx, info in layer_info.items():
            if idx in skip_layers:
                continue
                
            if info['type'] == 'Conv':
                conv = info['conv']
                mask = self.get_pruning_mask(conv, prune_ratio)
                masks[f'layer_{idx}'] = mask
                
                kept = mask.sum()
                total = len(mask)
                print(f"Layer {idx} (Conv): {kept}/{total} canais mantidos ({kept/total*100:.1f}%)")
        
        print("=" * 60)
        return masks
    
    def apply_pruning(self, prune_ratio=0.3, skip_layers=None):
        """
        Aplica pruning ao modelo reconstruindo as camadas
        
        Args:
            prune_ratio: Percentual de canais a remover (0-1)
            skip_layers: Lista de camadas a não fazer pruning
            
        Returns:
            Estatísticas do pruning
        """
        if skip_layers is None:
            # Não fazer pruning nas últimas camadas (detection head)
            skip_layers = list(range(len(self.original_model.model) - 3, len(self.original_model.model)))
        
        print(f"\nIniciando pruning estruturado...")
        print(f"Camadas ignoradas: {skip_layers}")
        
        # Computa máscaras de pruning
        self.pruning_masks = self.compute_pruning_plan(prune_ratio, skip_layers)
        
        # Conta parâmetros antes
        params_before = sum(p.numel() for p in self.original_model.parameters())
        
        # Aplica pruning reconstruindo camadas
        print("\nReconstruindo modelo com pruning...")
        self._rebuild_model_with_pruning(skip_layers)
        
        # Conta parâmetros depois
        params_after = sum(p.numel() for p in self.original_model.parameters())
        
        reduction = (1 - params_after / params_before) * 100
        
        stats = {
            'params_before': params_before,
            'params_after': params_after,
            'reduction_percent': reduction
        }
        
        print(f"\n✓ Pruning aplicado com sucesso!")
        print(f"  Parâmetros: {params_before:,} → {params_after:,}")
        print(f"  Redução: {reduction:.2f}%")
        
        return stats
    
    def _rebuild_model_with_pruning(self, skip_layers):
        """
        Reconstrói o modelo aplicando as máscaras de pruning
        """
        prev_mask = None
        
        for i, module in enumerate(self.original_model.model):
            if i in skip_layers:
                prev_mask = None
                continue
            
            mask_key = f'layer_{i}'
            
            if isinstance(module, Conv) and mask_key in self.pruning_masks:
                curr_mask = self.pruning_masks[mask_key]
                
                # Reconstrói Conv
                new_conv = self.prune_conv_layer(module.conv, prev_mask, curr_mask)
                module.conv = new_conv
                
                # Atualiza BatchNorm se existir
                if hasattr(module, 'bn'):
                    new_bn = self._prune_batchnorm(module.bn, curr_mask)
                    module.bn = new_bn
                
                prev_mask = curr_mask
            else:
                prev_mask = None
    
    def _prune_batchnorm(self, bn_layer, mask):
        """
        Cria nova camada BatchNorm com canais reduzidos
        """
        num_features = int(mask.sum())
        
        new_bn = nn.BatchNorm2d(num_features).to(self.device)
        
        # Copia parâmetros filtrados
        old_weight = bn_layer.weight.data.cpu().numpy()
        old_bias = bn_layer.bias.data.cpu().numpy()
        old_mean = bn_layer.running_mean.data.cpu().numpy()
        old_var = bn_layer.running_var.data.cpu().numpy()
        
        new_bn.weight.data = torch.from_numpy(old_weight[mask]).to(self.device)
        new_bn.bias.data = torch.from_numpy(old_bias[mask]).to(self.device)
        new_bn.running_mean.data = torch.from_numpy(old_mean[mask]).to(self.device)
        new_bn.running_var.data = torch.from_numpy(old_var[mask]).to(self.device)
        
        return new_bn
    
    def fine_tune(self, data_yaml, epochs=10, imgsz=640, batch=16):
        """
        Fine-tuning do modelo após pruning
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
        """
        return sum(p.numel() for p in self.original_model.parameters())
    
    def save_pruned_model(self, output_path):
        """
        Salva o modelo com pruning
        """
        self.model.save(output_path)
        print(f"\n✓ Modelo salvo em: {output_path}")


# Exemplo de uso
if __name__ == "__main__":
    # Configurações
    MODEL_PATH = "yolo11x.pt"
    DATA_YAML = "data.yaml"
    PRUNE_RATIO = 0.5  # 50% de redução
    FINETUNE_EPOCHS = 10
    OUTPUT_PATH = "yolo11x_pruned.pt"
    
    # Criar pruner
    pruner = YOLOPruner(MODEL_PATH)
    
    # Informações do modelo original
    print("\n" + "=" * 60)
    print("MODELO ORIGINAL")
    print("=" * 60)
    original_params = pruner.count_parameters()
    original_size = pruner.get_model_size()
    print(f"Parâmetros: {original_params:,}")
    print(f"Tamanho: {original_size:.2f} MB")
    
    # Avaliar modelo original
    original_metrics = pruner.evaluate(DATA_YAML)
    
    # Aplicar pruning
    print("\n" + "=" * 60)
    print("APLICANDO PRUNING")
    print("=" * 60)
    pruning_stats = pruner.apply_pruning(prune_ratio=PRUNE_RATIO)
    
    # Informações do modelo com pruning
    print("\n" + "=" * 60)
    print("MODELO COM PRUNING (antes do fine-tuning)")
    print("=" * 60)
    pruned_params = pruner.count_parameters()
    pruned_size = pruner.get_model_size()
    print(f"Parâmetros: {pruned_params:,}")
    print(f"Tamanho: {pruned_size:.2f} MB")
    print(f"Redução: {(1 - pruned_params/original_params)*100:.2f}%")
    
    # Avaliar modelo após pruning (antes do fine-tuning)
    print("\nAvaliando modelo após pruning (antes do fine-tuning)...")
    after_prune_metrics = pruner.evaluate(DATA_YAML)
    
    # Fine-tuning
    print("\n" + "=" * 60)
    print("FINE-TUNING")
    print("=" * 60)
    pruner.fine_tune(DATA_YAML, epochs=FINETUNE_EPOCHS, batch=16)
    
    # Avaliação final
    print("\n" + "=" * 60)
    print("AVALIAÇÃO FINAL")
    print("=" * 60)
    final_metrics = pruner.evaluate(DATA_YAML)
    
    # Salvar modelo
    pruner.save_pruned_model(OUTPUT_PATH)
    
    # Resumo completo
    print("\n" + "=" * 60)
    print("RESUMO COMPLETO")
    print("=" * 60)
    print(f"Parâmetros:")
    print(f"  Original:     {original_params:,}")
    print(f"  Com Pruning:  {pruned_params:,}")
    print(f"  Redução:      {(1 - pruned_params/original_params)*100:.2f}%")
    print(f"\nTamanho do modelo:")
    print(f"  Original:     {original_size:.2f} MB")
    print(f"  Com Pruning:  {pruned_size:.2f} MB")
    print(f"  Redução:      {(1 - pruned_size/original_size)*100:.2f}%")
    print(f"\nmAP50:")
    print(f"  Original:           {original_metrics.box.map50:.4f}")
    print(f"  Após Pruning:       {after_prune_metrics.box.map50:.4f} (Δ {after_prune_metrics.box.map50 - original_metrics.box.map50:+.4f})")
    print(f"  Após Fine-tuning:   {final_metrics.box.map50:.4f} (Δ {final_metrics.box.map50 - original_metrics.box.map50:+.4f})")
    print("=" * 60)