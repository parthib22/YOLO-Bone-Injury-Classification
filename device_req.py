import torch

print("GPU available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    "GPU Name: ",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
)
