import os
import torch
import pickle
import ai_edge_torch
from tqdm import tqdm
from torchvision import models
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet50
from torchvision.models import mobilenet_v2
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import VisionTransformer

list_torch_models = [
    resnet18,
    resnet50,
    mobilenet_v2,
    efficientnet_b0,
    vit_b_16,
]

list_input_shapes = [
    (224, 224),
    (256, 256),
    (384, 384),
    (512, 512),
]

save_dir = "outputs_tflite_fp32"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

for torch_model in tqdm(list_torch_models):
    for input_shape in tqdm(list_input_shapes):
        # Load the model
        model = torch_model(pretrained=False)
        model.eval()

        if isinstance(model, VisionTransformer) and input_shape[0] > 224:
            print(f'skip {torch_model}.... for {input_shape}')
            continue

        # Example input (replace with your actual input)
        x = torch.randn(1, 3, *input_shape)
        print(x.shape, x.dtype)
        sample_inputs = (x,)

        # Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
        # are setting the model to evaluation mode prior to conversion.
        edge_model = ai_edge_torch.convert(model, sample_inputs)
        tflite_path = f"{save_dir}/torch_{str(torch_model).split()[1]}_{input_shape[0]}.tflite"
        edge_model.export(tflite_path)
