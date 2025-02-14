import os
import torch
import torchvision.models as models

# Load the pretrained ViT-B/16 model
model = models.vit_b_16(pretrained=False)
model.eval()

# Create dummy input (ViT expects 224x224 images)
dummy_input = torch.randn(1, 3, 224, 224)

save_dir = "outputs_onnx"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Export model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    f=f"{save_dir}/vit_b16.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=15,  # Ensure compatibility
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("ONNX model saved as vit_b16.onnx")

# ./pnnx vit_b16.onnx inputshape=[1,3,224,224]
