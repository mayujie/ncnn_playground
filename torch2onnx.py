import os
import torch
import torchvision.models as models

if __name__ == '__main__':
    list_int_float = ['float32', 'float16', 'int8', 'uint8']

    # FLOAT_OPTION = "float16"
    FLOAT_OPTION = "float32"

    # Load the pretrained ViT-B/16 model
    model = models.vit_b_16(pretrained=False)
    model.eval()

    # Convert model to FP16
    if FLOAT_OPTION == "float16":
        model.half()

    # Create dummy input (ViT expects 224x224 images)
    dummy_input = torch.randn(1, 3, 224, 224).half() if FLOAT_OPTION == "float16" else torch.randn(1, 3, 224, 224)
    # dummy_input = torch.randn(1, 3, 224, 224)

    save_dir = "models_ncnn/outputs_onnx_fp16"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Export model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        f=f"{save_dir}/vit_b16_{FLOAT_OPTION}.onnx",
        input_names=["input"], output_names=["output"],

        opset_version=14,  # Ensure compatibility
        # opset_version=15,  # Ensure compatibility
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print("ONNX model saved as vit_b16.onnx")

    # ./pnnx vit_b16.onnx inputshape=[1,3,224,224]
    # ./pnnx vit_b16_float16.onnx inputshape=[1,3,224,224]f16
    # pnnx vit_b16_float16.onnx "inputshape=[1,3,224,224]"
    # pnnx vit_b16_float16.onnx "inputshape=[1,3,224,224]f16"
