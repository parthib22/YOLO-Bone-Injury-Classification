# !pip install roboflow

# from roboflow import Roboflow

# rf = Roboflow(api_key="p4j33hpzHmY3bWLP2F6O")
# project = rf.workspace("curso-rphcb").project("bone-break-classification")
# version = project.version(2)
# dataset = version.download("folder")

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("bone-vision").project("classification-itfyo")
version = project.version(2)
dataset = version.download("folder")
