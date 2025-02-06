import os
import torch
import pickle
import ai_edge_torch
from torchvision.models import efficientnet_b0

# Load the EfficientNet-B0 model
model = efficientnet_b0(pretrained=True)
model.eval()

# Example input (replace with your actual input)
x = torch.randn(1, 3, 224, 224)
sample_inputs = (x,)

save_dir = "outputs_tflite"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
# are setting the model to evaluation mode prior to conversion.
edge_model = ai_edge_torch.convert(model, sample_inputs)
edge_model.export(f"{save_dir}/{model._get_name()}.tflite")
