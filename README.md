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

### ncnn PTQ

https://github.com/Tencent/ncnn/tree/master/tools/quantize

## 2.pnnx

[pnnx (PyTorch Neural Network eXchange)](https://github.com/pnnx/pnnx)

[use ncnn with pytorch or onnx (how to use pnnx)](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx#how-to-use-pnnx)

[pnnx documentation read more](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)

## 3.onnx-simplifier & onnxsim

[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)

## Reference issues

pnnx

- https://github.com/pnnx/pnnx/issues/14 use bash
- https://github.com/pnnx/pnnx/issues/44 pnnx convert command
- https://github.com/Tencent/ncnn/issues/5570 pnnx convert

ncnn

- https://github.com/Tencent/ncnn/issues/2070 benchncnn benchmark 1
- https://github.com/Tencent/ncnn/issues/3662 benchncnn benchmark 2

tflite

- [tflite benchmark tools](https://ai.google.dev/edge/litert/models/measurement)

[Armv8 Neon technology](https://developer.arm.com/documentation/102474/0100/Fundamentals-of-Armv8-Neon-technology)

[cmake argument CPU architecture](https://github.com/Tencent/ncnn/blob/23890900c2a92a0932eba629d3c0bdbbc20808de/CMakeLists.txt#L267)