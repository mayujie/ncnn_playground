# ncnn_playground

The repository which play with ncnn framework

## 1.ncnn

[python wrapper of ncnn](https://github.com/Tencent/ncnn/tree/master/python)

### Download & Build status

[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build)

[build ncnn library samples](https://github.com/Tencent/ncnn/blob/master/build.sh)

[build-android ncnn library samples](https://github.com/Tencent/ncnn/blob/master/build-android.cmd)

```commandline
sudo apt install build-essential 
sudo apt install git 
sudo apt install cmake 
sudo apt install libprotobuf-dev 
sudo apt install protobuf-compiler 
sudo apt install libomp-dev 
sudo apt install libvulkan-dev 
sudo apt install vulkan-tools 

sudo apt install libopencv-dev
```

### ncnn benchmark

[ncnn benchmark](https://github.com/Tencent/ncnn/tree/master/benchmark)
adb command access to device
```
adb kill-server
adb connect <ip_address>:5555
adb shell 

cd /data/local/tmp/ 
adb push benchncnn /data/local/tmp/
adb push *.param /data/local/tmp/
adb push run_ncnn_benchmark.sh /data/local/tmp/
```

using my customized script
```
## original command
./benchncnn 500 1 0 0 0 param=vit_b16.ncnn.param shape=[224,224,3,1]

## GPU
./run_ncnn_benchmark.sh 500 1 0 0 0 vit_b16.ncnn.param   

## CPU thds=1
./run_ncnn_benchmark.sh 500 1 0 -1 0 torch_mobilenet_v2.ncnn.param                                                                                                              

## CPU thds=2
./run_ncnn_benchmark.sh 500 2 0 -1 0 torch_efficientnet_b0.ncnn.param  

## CPU thds=4
./run_ncnn_benchmark.sh 500 4 0 -1 0 torch_resnet50.ncnn.param                                                                                                                  

## CPU thds=8
./run_ncnn_benchmark.sh 500 8 0 -1 0 torch_resnet18.ncnn.param         
```

### ncnn PTQ

https://github.com/Tencent/ncnn/tree/master/tools/quantize

https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md

## 2.pnnx

pnnx convert command: 
```
pnnx vit_b16.onnx "inputshape=[1,3,224,224]"

pnnx vit_b16_float16.onnx "inputshape=[1,3,224,224]f16"
```

[pnnx (PyTorch Neural Network eXchange)](https://github.com/pnnx/pnnx)

[use ncnn with pytorch or onnx (how to use pnnx)](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx#how-to-use-pnnx)

[pnnx documentation read more](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)

## 3.onnx-simplifier & onnxsim

Extra: If Model Has Unsupported Operators

If `pnnx` throws an error about unsupported operations:

1. Try a lower ONNX opset version (e.g., `opset_version=9` or `opset_version=15` in the PyTorch export).
2. Manually modify the ONNX model using `onnx-simplifier`:
```
pip install onnx-simplifier

python -m onnxsim vit_b16.onnx vit_b16_simplified.onnx 

pnnx vit_b16_simplified.onnx "inputshape=[1,3,224,224]"
```
3. Use pnnx debugging mode to find issues:
```
pnnx vit_b16.onnx inputshape=[1,3,224,224] debug=1
```

[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)

4. Convert the FP32 ONNX Model to FP16 ONNX Model

    [Create Float16 and Mixed Precision Models](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html) `pip install onnx onnxconverter-common`
    ```
    import onnx
    from onnxconverter_common import float16
    
    model = onnx.load("path/to/model.onnx")
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "path/to/model_fp16.onnx")
    
    ```

## Reference issues

### pnnx

- https://github.com/pnnx/pnnx/issues/14 use bash
- https://github.com/pnnx/pnnx/issues/44 pnnx convert command
- https://github.com/Tencent/ncnn/issues/5570 pnnx convert

### ncnn

- https://github.com/Tencent/ncnn/issues/2070 benchncnn benchmark 1
- https://github.com/Tencent/ncnn/issues/3662 benchncnn benchmark 2

### tflite

- [tflite benchmark tools](https://ai.google.dev/edge/litert/models/measurement)

### ai-edge-torch
- https://github.com/google-ai-edge/ai-edge-torch/issues/150
- https://github.com/google-ai-edge/ai-edge-torch/blob/main/ai_edge_torch/generative/quantize/quant_recipes.py

[Armv8 Neon technology](https://developer.arm.com/documentation/102474/0100/Fundamentals-of-Armv8-Neon-technology)

[cmake argument CPU architecture](https://github.com/Tencent/ncnn/blob/23890900c2a92a0932eba629d3c0bdbbc20808de/CMakeLists.txt#L267)