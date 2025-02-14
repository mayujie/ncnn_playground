import os
import torch
import pickle
import ai_edge_torch
from torchvision import models
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet50
from torchvision.models import mobilenet_v2
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.models.vision_transformer import vit_b_16

list_torch_models = [
    resnet18,
    resnet50,
    mobilenet_v2,
    efficientnet_b0,
    vit_b_16
]

save_dir = "outputs_tflite"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

for torch_model in list_torch_models:
    # Load the EfficientNet-B0 model
    model = torch_model(pretrained=False)
    model.eval()

    # Example input (replace with your actual input)
    x = torch.randn(1, 3, 224, 224)
    sample_inputs = (x,)

    # Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
    # are setting the model to evaluation mode prior to conversion.
    edge_model = ai_edge_torch.convert(model, sample_inputs)
    edge_model.export(f"{save_dir}/torch_{str(torch_model).split()[1]}.tflite")
