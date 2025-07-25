from collections import defaultdict
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

message = f"Confidence Values for | {source} |"
divs = "-" * max(40, len(message) + 1)

print(f"\n{divs}\n{message}\n{divs}")

results = model.predict(source=source, conf=0.25, device=device, verbose=False)

preds = defaultdict(float)

count = len(results)

for result in results:

    try:
        datas = result.probs.data.cpu().numpy()
        names = result.names.values()
    except Exception as e:
        print(f"Error processing result: {e}")

    for data, name in zip(datas, names):
        preds[name] += float(data)

for name, data in sorted(list(preds.items()), key=lambda x: x[1], reverse=True):
    print(f"{name} ----- {data/count*100:.2f}%")
