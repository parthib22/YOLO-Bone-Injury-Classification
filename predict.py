import torch
import numpy as np
import os
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

"""
['Avulsion fracture', 'Comminuted fracture', 'Dislocation Fracture',
 'Fracture Hairline', 'Fracture Spiral', 'Greenstick fracture',
 'Impacted fracture', 'Longitudinal fracture', 'Oblique fracture',
 'Pathological fracture']
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# user = input("Folder or file ----- ")
# source = r"{}".format(user) if user.startswith("C:\\") else user
source = "content"
if not os.path.exists(source):
    print(f"Source '{source}' does not exist. Please check the path.")
    exit(1)

results = model.predict(source=source, conf=0.25, device=device, verbose=False)

# print(len(results))
for idx, filename in enumerate(os.listdir(source)):

    preds = []

    try:
        datas = results[idx].probs.data.cpu().numpy()
        names = results[idx].names.values()
    except Exception as e:
        print(f"Error processing result: {e}")

    for data, name in zip(datas, names):
        preds.append((name, float(data) * 100))

    preds.sort(key=lambda x: x[1], reverse=True)

    message = f"Confidence Values for | {filename} |"
    divs = "-" * max(40, len(message) + 1)

    # cv2.imshow("X-Ray", cv2.imread(f"content/{filename}"))
    print(f"\n{divs}\n{message}\n{divs}")

    for pred in preds[:3]:
        print(f"{pred[0]} ----- {pred[1]:.2f}%")
