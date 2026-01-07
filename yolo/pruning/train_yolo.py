from ultralytics import YOLO
import torch

num_epochs = 100
imgsz = 640

model = YOLO("yolo11x.pt")


model.train(
        data="./data.yaml",
        workers=4,
        epochs=num_epochs,
        imgsz=imgsz,
        batch=16,
        device=0 ,
        project="runs/train",
        name="yolo11x__oxford_tower_custom_train"
    )

print("Training finished.")
#temp

   


