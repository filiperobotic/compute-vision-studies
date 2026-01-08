import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
import copy

class YOLOPruner:
    """
    Pruning estruturado para YOLOv11 usando torch.nn.utils.prune
    """
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        self.original_state = copy.deepcopy(self.model.model.state_dict())
        
    def calculate_channel_importance(self, module):
        """
        Calcula importância de cada canal usando norma L1
        """
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            # Importância = soma absoluta dos pesos de cada filtro
            importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
            return importance.cpu().numpy()
        return None
    
    def apply_structured_pruning(self, prune_ratio=0.3, skip_layers=None):
        """
        Aplica pruning estruturado zerando canais menos importantes
        
        Args:
            prune_ratio: Percentual de canais a zerar
            skip_layers: Lista de nomes de camadas a ignorar
        """
        if skip_layers is None:
            skip_layers = []
        
        print(f"\n{'='*60}")
        print(f"APLICANDO PRUNING ESTRUTURADO (Ratio: {prune_ratio:.1%})")
        print(f"{'='*60}")
        
        total_params_before = sum(p.numel() for p in self.model.model.parameters())
        pruned_count = 0
        
        # Aplica pruning a cada camada Conv2d
        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Verifica se deve pular esta camada
                should_skip = any(skip in name for skip in skip_layers)
                
                # Não fazer pruning em camadas muito pequenas ou do detection head
                if should_skip or module.out_channels <= 16 or 'detect' in name.lower():
                    continue
                
                importance = self.calculate_channel_importance(module)
                if importance is None:
                    continue
                
                num_channels = len(importance)
                num_prune = int(num_channels * prune_ratio)
                
                if num_prune == 0:
                    continue
                
                # Índices dos canais menos importantes
                threshold_idx = np.argsort(importance)[num_prune]
                threshold = importance[threshold_idx]
                
                # Cria máscara (0 para canais fracos, 1 para fortes)
                mask = (importance > threshold).astype(np.float32)
                mask_tensor = torch.from_numpy(mask).to(self.device)
                
                # Aplica máscara aos pesos (zera canais fracos)
                with torch.no_grad():
                    module.weight.data *= mask_tensor.view(-1, 1, 1, 1)
                    if module.bias is not None:
                        module.bias.data *= mask_tensor
                
                kept = int(mask.sum())
                pruned_count += 1
                print(f"  {name:40s} → {kept:3d}/{num_channels:3d} canais ({kept/num_channels*100:5.1f}%)")
        
        total_params_after = sum(p.numel() for p in self.model.model.parameters())
        
        # Calcula esparsidade real (quantos pesos são zero)
        zero_params = sum((p == 0).sum().item() for p in self.model.model.parameters())
        sparsity = zero_params / total_params_after * 100
        
        print(f"\n{'='*60}")
        print(f"✓ Pruning aplicado em {pruned_count} camadas")
        print(f"  Parâmetros totais: {total_params_after:,}")
        print(f"  Parâmetros zerados: {zero_params:,} ({sparsity:.2f}%)")
        print(f"{'='*60}\n")
        
        return {
            'pruned_layers': pruned_count,
            'total_params': total_params_after,
            'zero_params': zero_params,
            'sparsity': sparsity
        }
    
    def iterative_pruning(self, target_sparsity=0.5, steps=5, data_yaml=None, 
                          finetune_epochs=5, skip_layers=None):
        """
        Pruning iterativo com fine-tuning entre etapas
        
        Args:
            target_sparsity: Esparsidade final desejada (0-1)
            steps: Número de etapas de pruning
            data_yaml: Dataset para fine-tuning
            finetune_epochs: Épocas de fine-tuning por etapa
            skip_layers: Camadas a não fazer pruning
        """
        print(f"\n{'='*60}")
        print(f"PRUNING ITERATIVO")
        print(f"  Target sparsity: {target_sparsity:.1%}")
        print(f"  Steps: {steps}")
        print(f"  Fine-tune epochs per step: {finetune_epochs}")
        print(f"{'='*60}")
        
        # Calcula ratio por etapa (gradual pruning)
        prune_ratio_per_step = 1 - (1 - target_sparsity) ** (1/steps)
        
        for step in range(steps):
            print(f"\n{'='*60}")
            print(f"STEP {step+1}/{steps}")
            print(f"{'='*60}")
            
            # Aplica pruning
            stats = self.apply_structured_pruning(
                prune_ratio=prune_ratio_per_step,
                skip_layers=skip_layers
            )
            
            # Fine-tuning se dataset fornecido
            if data_yaml and finetune_epochs > 0:
                print(f"\nFine-tuning por {finetune_epochs} épocas...")
                self.model.train(
                    data=data_yaml,
                    epochs=finetune_epochs,
                    batch=16,
                    device=self.device,
                    verbose=False,
                    patience=10
                )
                
                # Avalia
                metrics = self.evaluate(data_yaml)
                print(f"mAP50 após step {step+1}: {metrics.box.map50:.4f}")
        
        return stats
    
    def fine_tune(self, data_yaml, epochs=30, imgsz=640, batch=16, patience=20):
        """
        Fine-tuning do modelo após pruning
        """
        print(f"\n{'='*60}")
        print(f"FINE-TUNING FINAL")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}, Batch: {batch}, Patience: {patience}")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            patience=patience,
            save=True,
            verbose=True,
            plots=True,
            project='runs/prune',
            name='finetune'
        )
        return results
    
    def evaluate(self, data_yaml, imgsz=640, verbose=False):
        """
        Avalia o modelo
        """
        if verbose:
            print("\nAvaliando modelo...")
        
        metrics = self.model.val(
            data=data_yaml,
            imgsz=imgsz,
            device=self.device,
            verbose=verbose,
            plots=False
        )
        return metrics
    
    def remove_pruning_reparameterization(self):
        """
        Remove os hooks de pruning e torna as máscaras permanentes
        """
        for module in self.model.model.modules():
            if isinstance(module, nn.Conv2d):
                # Remove reparametrização do pruning se existir
                if hasattr(module, 'weight_orig'):
                    # Torna os pesos "pruned" permanentes
                    module.weight = nn.Parameter(module.weight.data)
                    del module.weight_orig
                    del module.weight_mask
    
    def export_sparse_model(self, output_path):
        """
        Exporta modelo esparso (com zeros)
        """
        self.remove_pruning_reparameterization()
        self.model.save(output_path)
        print(f"\n✓ Modelo esparso salvo: {output_path}")
    
    def get_model_info(self):
        """
        Retorna informações sobre o modelo
        """
        total_params = sum(p.numel() for p in self.model.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.model.parameters())
        nonzero_params = total_params - zero_params
        sparsity = zero_params / total_params * 100
        
        # Calcula tamanho em MB
        param_size = sum(p.nelement() * p.element_size() 
                        for p in self.model.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() 
                         for b in self.model.model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'nonzero_params': nonzero_params,
            'sparsity': sparsity,
            'size_mb': size_mb
        }
    
    def save_pruned_model(self, output_path):
        """
        Salva modelo com pruning
        """
        self.model.save(output_path)
        print(f"\n✓ Modelo salvo: {output_path}")


# Exemplo de uso
if __name__ == "__main__":
    # Configurações
    MODEL_PATH = "yolo11x.pt"
    DATA_YAML = "data.yaml"
    PRUNE_RATIO = 0.5  # 50% de esparsidade
    FINETUNE_EPOCHS = 50
    OUTPUT_PATH = "yolo11x_pruned.pt"
    
    print("="*70)
    print("YOLO PRUNING - Pruning Estruturado com Fine-tuning")
    print("="*70)
    
    # Criar pruner
    pruner = YOLOPruner(MODEL_PATH)
    
    # Informações do modelo original
    print("\n" + "="*70)
    print("MODELO ORIGINAL")
    print("="*70)
    original_info = pruner.get_model_info()
    print(f"Parâmetros totais: {original_info['total_params']:,}")
    print(f"Tamanho: {original_info['size_mb']:.2f} MB")
    
    # Avaliar modelo original
    print("\n📊 Avaliando modelo original...")
    original_metrics = pruner.evaluate(DATA_YAML, verbose=True)
    original_map50 = original_metrics.box.map50
    print(f"mAP50: {original_map50:.4f}")
    
    # Método 1: Pruning único
    print("\n" + "="*70)
    print("MÉTODO 1: PRUNING ÚNICO")
    print("="*70)
    
    stats = pruner.apply_structured_pruning(
        prune_ratio=PRUNE_RATIO,
        skip_layers=['model.22']  # Pula detection head
    )
    
    # Informações após pruning
    print("\n" + "="*70)
    print("APÓS PRUNING (antes do fine-tuning)")
    print("="*70)
    after_prune_info = pruner.get_model_info()
    print(f"Parâmetros totais: {after_prune_info['total_params']:,}")
    print(f"Parâmetros não-zero: {after_prune_info['nonzero_params']:,}")
    print(f"Parâmetros zerados: {after_prune_info['zero_params']:,}")
    print(f"Esparsidade: {after_prune_info['sparsity']:.2f}%")
    print(f"Tamanho: {after_prune_info['size_mb']:.2f} MB")
    
    # Avaliar após pruning
    print("\n📊 Avaliando após pruning (antes do fine-tuning)...")
    after_prune_metrics = pruner.evaluate(DATA_YAML, verbose=True)
    after_prune_map50 = after_prune_metrics.box.map50
    print(f"mAP50: {after_prune_map50:.4f} (Δ {after_prune_map50 - original_map50:+.4f})")
    
    # Fine-tuning
    print("\n" + "="*70)
    print("FINE-TUNING")
    print("="*70)
    pruner.fine_tune(DATA_YAML, epochs=FINETUNE_EPOCHS, batch=16, patience=20)
    
    # Avaliação final
    print("\n" + "="*70)
    print("AVALIAÇÃO FINAL")
    print("="*70)
    final_metrics = pruner.evaluate(DATA_YAML, verbose=True)
    final_map50 = final_metrics.box.map50
    print(f"mAP50: {final_map50:.4f}")
    
    # Informações finais
    final_info = pruner.get_model_info()
    
    # Salvar modelo
    pruner.save_pruned_model(OUTPUT_PATH)
    
    # Resumo completo
    print("\n" + "="*70)
    print("📊 RESUMO COMPLETO")
    print("="*70)
    
    print(f"\n🔧 CONFIGURAÇÃO:")
    print(f"  Modelo original: {MODEL_PATH}")
    print(f"  Modelo pruned: {OUTPUT_PATH}")
    print(f"  Prune ratio: {PRUNE_RATIO:.1%}")
    
    print(f"\n📦 COMPRESSÃO:")
    print(f"  Tamanho original: {original_info['size_mb']:.2f} MB")
    print(f"  Tamanho final: {final_info['size_mb']:.2f} MB")
    print(f"  Redução: {(1 - final_info['size_mb']/original_info['size_mb'])*100:.2f}%")
    
    print(f"\n🔢 PARÂMETROS:")
    print(f"  Total original: {original_info['total_params']:,}")
    print(f"  Não-zero final: {final_info['nonzero_params']:,}")
    print(f"  Esparsidade: {final_info['sparsity']:.2f}%")
    print(f"  Redução efetiva: {(1 - final_info['nonzero_params']/original_info['total_params'])*100:.2f}%")
    
    print(f"\n📈 PERFORMANCE (mAP50):")
    print(f"  Original:          {original_map50:.4f}")
    print(f"  Após Pruning:      {after_prune_map50:.4f} ({after_prune_map50 - original_map50:+.4f})")
    print(f"  Após Fine-tuning:  {final_map50:.4f} ({final_map50 - original_map50:+.4f})")
    
    if after_prune_map50 < original_map50:
        recovery = (final_map50 - after_prune_map50) / (original_map50 - after_prune_map50) * 100
        print(f"  Recovery: {recovery:.1f}%")
    
    print("\n" + "="*70)
    print("✅ PROCESSO COMPLETO!")
    print("="*70)
    
    print("\n💡 PRÓXIMOS PASSOS:")
    print("  1. Teste o modelo pruned em inferência")
    print("  2. Para redução real de tamanho, use quantização:")
    print("     model.export(format='engine', half=True)  # TensorRT FP16")
    print("  3. Ou combine com Knowledge Distillation para melhor performance")