import os
import torch
import pnnx
from torchvision import models
from tqdm import tqdm
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
    # vit_b_16
]

USE_FP16 = True

save_dir = "outputs_ncnn_fp16_gpuhaha"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

for torch_model in tqdm(list_torch_models):
    model = torch_model(pretrained=False)
    # model.half()
    model.eval()

    x = torch.rand(1, 3, 224, 224)
    # x = torch.rand((1, 3, 224, 224), dtype=torch.float16)

    opt_model = pnnx.export(
        model,
        ptpath=os.path.join(
            os.path.dirname(__file__), save_dir,
            f"torch_{str(torch_model).split()[1]}_fp16.pt" if USE_FP16 else f"torch_{str(torch_model).split()[1]}.pt"),
        inputs=x,
        # device='gpu',
        fp16=USE_FP16,
        optlevel=2,
    )

    # use tuple for model with multiple inputs
    # opt_model = pnnx.export(model, "resnet18.pt", (x, y, z))

    result = opt_model(x)
    print(result)
