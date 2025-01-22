import os
import torch
import pnnx
from torchvision.models import efficientnet

model = efficientnet.efficientnet_b0(pretrained=False)

x = torch.rand(1, 3, 224, 224)

save_dir = "outputs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

opt_model = pnnx.export(model, f"{save_dir}/resnet18.pt", x)

# use tuple for model with multiple inputs
# opt_model = pnnx.export(model, "resnet18.pt", (x, y, z))

result = opt_model(x)
print()
