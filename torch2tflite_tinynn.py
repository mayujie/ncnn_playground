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
from tinynn.converter import TFLiteConverter

# Ensure PyTorch is using CPU only
torch.backends.cudnn.enabled = False  # Disable CuDNN
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices

list_torch_models = [
    resnet18,
    resnet50,
    mobilenet_v2,
    efficientnet_b0,
    # vit_b_16,
]

list_input_shapes = [
    (224, 224),
    (256, 256),
    (384, 384),
    (512, 512),
]

use_float16 = True
save_dir = "outputs_tflite_fp16_tinynn" if use_float16 else "outputs_tflite_fp32_tinynn"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

for torch_model in tqdm(list_torch_models):
    for input_shape in tqdm(list_input_shapes):
        # Load the model and enforce CPU usage
        # model = torch_model(pretrained=False)
        model = torch_model(pretrained=False).to("cpu")
        model.eval()

        if isinstance(model, VisionTransformer) and input_shape[0] > 224:
            print(f'skip {torch_model}.... for {input_shape}')
            continue

        suffix = 'float16' if use_float16 else 'float32'
        dummy_input = torch.randn((1, 3, *input_shape))
        print(dummy_input.shape, dummy_input.dtype)
        scripted_model = torch.jit.trace(model, dummy_input)

        tflite_path = f"{save_dir}/torch_{str(torch_model).split()[1]}_{input_shape[0]}_{suffix}.tflite"
        converter = TFLiteConverter(
            model=model,
            dummy_input=dummy_input,
            tflite_path=tflite_path,
            float16_quantization=use_float16,
            fuse_quant_dequant=True,
            # quantize_target_type='int8',
            # rewrite_quantizable=True,
        )
        converter.convert()
