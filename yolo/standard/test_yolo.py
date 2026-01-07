from ultralytics import YOLO

trained_model_path = '/home/pesquisador/pesquisa/filipe/compute-vision-studies/runs/train/yolo11x__oxford_tower_custom_train/weights/best.pt'

# Carrega o modelo treinado
model = YOLO(trained_model_path)

# Avalia o modelo utilizando o conjunto de teste
results = model.val(data='data.yaml', split='test')

# Exibe os principais resultados
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")