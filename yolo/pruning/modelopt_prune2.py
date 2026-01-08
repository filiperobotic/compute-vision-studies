import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from copy import deepcopy
import yaml

class YOLOPruner:
    """
    Classe para realizar pruning estruturado em modelos YOLOv11 via sparse training
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
        self.original_model_path = model_path
        self.model.to(device)
        
    def create_sparse_model_config(self, original_model, prune_ratio=0.3):
        """
        Cria configuração para modelo esparso baseado no original
        
        Args:
            original_model: Modelo YOLO original
            prune_ratio: Ratio de redução de canais
            
        Returns:
            Dicionário com nova configuração
        """
        # Extrai configuração do modelo original
        model_cfg = original_model.model.yaml
        
        # Calcula novo width_multiple
        original_width = model_cfg.get('width_multiple', 1.0)
        new_width = original_width * (1 - prune_ratio)
        
        # Cria nova configuração
        new_cfg = deepcopy(model_cfg)
        new_cfg['width_multiple'] = new_width
        
        return new_cfg
    
    def create_pruned_model(self, prune_ratio=0.3):
        """
        Cria um novo modelo com arquitetura reduzida
        
        Args:
            prune_ratio: Percentual de redução (0-1)
            
        Returns:
            Novo modelo YOLO com menos canais
        """
        print(f"\nCriando modelo com {prune_ratio*100:.0f}% de redução...")
        
        # Detecta o modelo base (n, s, m, l, x)
        model_name = self.original_model_path.split('/')[-1]
        
        # Mapa de modelos e seus width_multiple
        width_map = {
            'yolo11n': 0.25,
            'yolo11s': 0.50,
            'yolo11m': 0.75,
            'yolo11l': 1.0,
            'yolo11x': 1.25,
        }
        
        # Identifica o modelo base
        base_model = None
        for key in width_map.keys():
            if key in model_name.lower():
                base_model = key
                break
        
        if base_model is None:
            base_model = 'yolo11n'  # Fallback
        
        # Calcula novo width_multiple
        original_width = width_map[base_model]
        new_width = original_width * (1 - prune_ratio)
        
        # Carrega modelo base e modifica width
        from ultralytics.nn.tasks import DetectionModel
        
        # Cria configuração customizada
        cfg_dict = {
            'nc': self.model.model.nc,  # número de classes
            'scales': {
                'pruned': {
                    'depth': self.model.model.yaml.get('depth_multiple', 0.33),
                    'width': new_width
                }
            }
        }
        
        # Cria modelo com nova arquitetura
        new_model = YOLO(f'{base_model}.yaml')
        
        # Ajusta width manualmente
        self._scale_model_width(new_model.model, new_width)
        
        return new_model
    
    def _scale_model_width(self, model, width_scale):
        """
        Ajusta manualmente o width_multiple do modelo
        """
        if hasattr(model, 'yaml'):
            model.yaml['width_multiple'] = width_scale
    
    def transfer_weights(self, source_model, target_model):
        """
        Transfere pesos do modelo original para o modelo reduzido
        usando critério de importância (norma L1)
        
        Args:
            source_model: Modelo original (maior)
            target_model: Modelo reduzido (menor)
        """
        print("\nTransferindo pesos importantes do modelo original...")
        
        source_layers = list(source_model.model.modules())
        target_layers = list(target_model.model.modules())
        
        transferred = 0
        
        for src_module, tgt_module in zip(source_layers, target_layers):
            if isinstance(src_module, nn.Conv2d) and isinstance(tgt_module, nn.Conv2d):
                # Verifica se as dimensões são compatíveis para transferência parcial
                src_out = src_module.out_channels
                tgt_out = tgt_module.out_channels
                src_in = src_module.in_channels
                tgt_in = tgt_module.in_channels
                
                if tgt_out <= src_out and tgt_in <= src_in:
                    # Calcula importância dos filtros de saída
                    weight = src_module.weight.data.cpu().numpy()
                    importance_out = np.sum(np.abs(weight), axis=(1, 2, 3))
                    
                    # Seleciona os top-k filtros mais importantes
                    top_indices_out = np.argsort(importance_out)[-tgt_out:]
                    top_indices_out = np.sort(top_indices_out)
                    
                    # Para canais de entrada, calcula importância
                    importance_in = np.sum(np.abs(weight), axis=(0, 2, 3))
                    top_indices_in = np.argsort(importance_in)[-tgt_in:]
                    top_indices_in = np.sort(top_indices_in)
                    
                    # Transfere pesos selecionados
                    selected_weight = weight[top_indices_out][:, top_indices_in]
                    tgt_module.weight.data = torch.from_numpy(selected_weight).to(self.device)
                    
                    # Transfere bias se existir
                    if src_module.bias is not None and tgt_module.bias is not None:
                        selected_bias = src_module.bias.data.cpu().numpy()[top_indices_out]
                        tgt_module.bias.data = torch.from_numpy(selected_bias).to(self.device)
                    
                    transferred += 1
            
            elif isinstance(src_module, nn.BatchNorm2d) and isinstance(tgt_module, nn.BatchNorm2d):
                # Transfere BatchNorm
                src_features = src_module.num_features
                tgt_features = tgt_module.num_features
                
                if tgt_features <= src_features:
                    # Calcula importância baseada no peso do BatchNorm
                    bn_weight = src_module.weight.data.cpu().numpy()
                    importance = np.abs(bn_weight)
                    top_indices = np.argsort(importance)[-tgt_features:]
                    top_indices = np.sort(top_indices)
                    
                    # Transfere parâmetros
                    tgt_module.weight.data = torch.from_numpy(bn_weight[top_indices]).to(self.device)
                    
                    bn_bias = src_module.bias.data.cpu().numpy()
                    tgt_module.bias.data = torch.from_numpy(bn_bias[top_indices]).to(self.device)
                    
                    running_mean = src_module.running_mean.data.cpu().numpy()
                    tgt_module.running_mean.data = torch.from_numpy(running_mean[top_indices]).to(self.device)
                    
                    running_var = src_module.running_var.data.cpu().numpy()
                    tgt_module.running_var.data = torch.from_numpy(running_var[top_indices]).to(self.device)
        
        print(f"✓ Transferidos pesos de {transferred} camadas convolucionais")
    
    def apply_pruning(self, prune_ratio=0.3):
        """
        Aplica pruning criando novo modelo com arquitetura reduzida
        
        Args:
            prune_ratio: Percentual de redução (0-1)
            
        Returns:
            Estatísticas do pruning
        """
        print("\n" + "=" * 60)
        print("INICIANDO PRUNING ESTRUTURADO")
        print("=" * 60)
        
        # Conta parâmetros do modelo original
        original_model = self.model
        params_before = sum(p.numel() for p in original_model.model.parameters())
        
        # Cria modelo reduzido
        pruned_model = self.create_pruned_model(prune_ratio)
        
        # Transfere pesos importantes
        self.transfer_weights(original_model.model, pruned_model.model)
        
        # Substitui modelo
        self.model = pruned_model
        self.model.to(self.device)
        
        # Conta parâmetros do modelo reduzido
        params_after = sum(p.numel() for p in self.model.model.parameters())
        
        reduction = (1 - params_after / params_before) * 100
        
        stats = {
            'params_before': params_before,
            'params_after': params_after,
            'reduction_percent': reduction
        }
        
        print(f"\n✓ Pruning aplicado com sucesso!")
        print(f"  Parâmetros: {params_before:,} → {params_after:,}")
        print(f"  Redução: {reduction:.2f}%")
        print("=" * 60)
        
        return stats
    
    def fine_tune(self, data_yaml, epochs=30, imgsz=640, batch=16, patience=20):
        """
        Fine-tuning do modelo após pruning
        
        Args:
            data_yaml: Caminho para arquivo YAML do dataset
            epochs: Número de épocas
            imgsz: Tamanho da imagem
            batch: Tamanho do batch
            patience: Early stopping patience
        """
        print(f"\nIniciando fine-tuning por até {epochs} épocas...")
        print(f"Early stopping: {patience} épocas sem melhoria")
        
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
    
    def evaluate(self, data_yaml, imgsz=640):
        """
        Avalia o modelo
        """
        print("\nAvaliando modelo...")
        metrics = self.model.val(
            data=data_yaml,
            imgsz=imgsz,
            device=self.device,
            plots=False
        )
        return metrics
    
    def get_model_size(self):
        """
        Calcula o tamanho do modelo em MB
        """
        param_size = 0
        for param in self.model.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def count_parameters(self):
        """
        Conta o número de parâmetros do modelo
        """
        return sum(p.numel() for p in self.model.model.parameters())
    
    def save_pruned_model(self, output_path):
        """
        Salva o modelo com pruning
        """
        self.model.save(output_path)
        print(f"\n✓ Modelo salvo em: {output_path}")
    
    def export_onnx(self, output_path='model_pruned.onnx', imgsz=640):
        """
        Exporta modelo para ONNX
        """
        print(f"\nExportando para ONNX: {output_path}")
        self.model.export(format='onnx', imgsz=imgsz, dynamic=False)
        print("✓ Exportação concluída")


# Exemplo de uso
if __name__ == "__main__":
    # Configurações
    MODEL_PATH = "yolo11x.pt"
    DATA_YAML = "data.yaml"
    PRUNE_RATIO = 0.30  # 30% de redução
    FINETUNE_EPOCHS = 30
    OUTPUT_PATH = "yolo11x_pruned.pt"
    
    print("=" * 60)
    print("YOLO PRUNING - Redução de Modelo via Arquitetura")
    print("=" * 60)
    
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
    print("\n📊 Avaliando modelo original...")
    original_metrics = pruner.evaluate(DATA_YAML)
    original_map50 = original_metrics.box.map50
    print(f"mAP50: {original_map50:.4f}")
    
    # Aplicar pruning
    pruning_stats = pruner.apply_pruning(prune_ratio=PRUNE_RATIO)
    
    # Informações do modelo com pruning
    print("\n" + "=" * 60)
    print("MODELO COM PRUNING (antes do fine-tuning)")
    print("=" * 60)
    pruned_params = pruner.count_parameters()
    pruned_size = pruner.get_model_size()
    print(f"Parâmetros: {pruned_params:,}")
    print(f"Tamanho: {pruned_size:.2f} MB")
    print(f"Redução de parâmetros: {(1 - pruned_params/original_params)*100:.2f}%")
    print(f"Redução de tamanho: {(1 - pruned_size/original_size)*100:.2f}%")
    
    # Avaliar modelo após pruning (antes do fine-tuning)
    print("\n📊 Avaliando modelo após pruning (antes do fine-tuning)...")
    after_prune_metrics = pruner.evaluate(DATA_YAML)
    after_prune_map50 = after_prune_metrics.box.map50
    print(f"mAP50: {after_prune_map50:.4f} (Δ {after_prune_map50 - original_map50:+.4f})")
    
    # Fine-tuning
    print("\n" + "=" * 60)
    print("FINE-TUNING")
    print("=" * 60)
    print("Isso pode levar algum tempo...")
    pruner.fine_tune(DATA_YAML, epochs=FINETUNE_EPOCHS, batch=16, patience=20)
    
    # Avaliação final
    print("\n" + "=" * 60)
    print("AVALIAÇÃO FINAL")
    print("=" * 60)
    final_metrics = pruner.evaluate(DATA_YAML)
    final_map50 = final_metrics.box.map50
    print(f"mAP50: {final_map50:.4f}")
    
    # Salvar modelo
    pruner.save_pruned_model(OUTPUT_PATH)
    
    # Exportar para ONNX (opcional)
    # pruner.export_onnx('model_pruned.onnx')
    
    # Resumo completo
    print("\n" + "=" * 60)
    print("📊 RESUMO COMPLETO")
    print("=" * 60)
    print(f"\n🔧 MODELO:")
    print(f"  Original:      {MODEL_PATH}")
    print(f"  Pruned:        {OUTPUT_PATH}")
    print(f"  Prune Ratio:   {PRUNE_RATIO*100:.0f}%")
    
    print(f"\n📦 TAMANHO:")
    print(f"  Original:      {original_size:.2sf} MB")
    print(f"  Pruned:        {pruned_size:.2f} MB")
    print(f"  Redução:       {(1 - pruned_size/original_size)*100:.2f}%")
    
    print(f"\n🔢 PARÂMETROS:")
    print(f"  Original:      {original_params:,}")
    print(f"  Pruned:        {pruned_params:,}")
    print(f"  Redução:       {(1 - pruned_params/original_params)*100:.2f}%")
    
    print(f"\n📈 PERFORMANCE (mAP50):")
    print(f"  Original:              {original_map50:.4f}")
    print(f"  Após Pruning:          {after_prune_map50:.4f} ({after_prune_map50 - original_map50:+.4f})")
    print(f"  Após Fine-tuning:      {final_map50:.4f} ({final_map50 - original_map50:+.4f})")
    print(f"  Recovery:              {((final_map50 - after_prune_map50)/(original_map50 - after_prune_map50)*100):.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ PROCESSO COMPLETO!")
    print("=" * 60)