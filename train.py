from ultralytics import YOLO

model = YOLO(
    "yolov8n-cls.pt"
)  # use a pre-trained model, find more at https://docs.ultralytics.com/tasks/classify/

data_yaml_path = (
    "Bone-Break-Classification-2"  # Update with your dataset YAML file path
)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model.train(data=data_yaml_path, epochs=50, device=device)
