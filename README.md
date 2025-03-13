# ncnn_playground

The repository which play with ncnn framework

## 1.ncnn

[python wrapper of ncnn](https://github.com/Tencent/ncnn/tree/master/python)

### Download & Build status

[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build)
```
cd <ncnn-root-dir>
mkdir -p build-android-aarch64
cd build-android-aarch64

export ANDROID_NDK=/home/yujiema/project/ncnn_playground/pre_built_ncnn/android-ndk-r27c
source ~/.bashrc
source ~/.zshrc

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-34 \
    -DNCNN_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    ..

make -j$(nproc)
make install
```

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

adb pull /data/local/tmp/result_benchmark.ncnn.log .
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

https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnnoptimize-to-optimize-model.md

### ncnn Quant 中文
1. **[Ncnn int8化](https://www.cnblogs.com/ayou27/p/16266497.html)**
2. [ncnn INT8量化实战指南：利用百度智能云文心快码（Comate）优化深度学习模型](https://cloud.baidu.com/article/3321906)
3. [ncnn 模型转换与量化：从理论到实践](https://cloud.baidu.com/article/3322108)
4. [ncnn发布20210507版本，int8量化推理大幅优化超500%](https://baijiahao.baidu.com/s?id=1699724039745016586&wfr=spider&for=pc)
5. **[ncnn发布20210507版本，int8量化推理大优化超500% zhihu](https://zhuanlan.zhihu.com/p/370689914)**

### [ncnn系列](https://so.csdn.net/so/search?q=ncnn&t=blog&u=shanglianlm)

- **[ncnn之二：Linux环境下ncnn安装+protobuf+opencv](https://blog.csdn.net/shanglianlm/article/details/103188992)**
- [ncnn之六：ncnn量化(post-training quantization)三部曲 - ncnnoptimize](https://blog.csdn.net/shanglianlm/article/details/103746080)
- [ncnn之七：ncnn量化(post-training quantization)三部曲 - ncnn2table](https://blog.csdn.net/shanglianlm/article/details/103745674)
- https://developer.baidu.com/article/details/2969970
- [ncnn之八：ncnn量化(post-training quantization)三部曲 - ncnn2int8](https://blog.csdn.net/shanglianlm/article/details/103745975)

**[ncnn模型 int8量化](https://blog.csdn.net/flyfish1986/article/details/131411144)**

[使用NCNN的INT8量化方式进行推理](https://blog.csdn.net/tugouxp/article/details/122489836)

[实战解析：如何使用NCNN进行INT8量化与部署优化](https://cloud.baidu.com/article/3321975)

**[NCNN深度学习框架之Optimize优化器](https://www.cnblogs.com/wanggangtao/p/11313705.html)**

**[NCNN量化之ncnn2table和ncnn2int8](https://www.cnblogs.com/wanggangtao/p/11352948.html)**

**[ncnn框架量化工具过程记录笔记](https://zhuanlan.zhihu.com/p/362701667)**

**[ncnn框架编译和量化](https://zhuanlan.zhihu.com/p/543666918)**

----
[必看部署系列~懂你的神经网络量化教程：第一讲！](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247488318&idx=1&sn=048c1b78f3b2cb25c05abb115f20d6c6&chksm=cf108b3bf867022d1b214928102d65ed691c81955b59ca02bccdee92584ad9aa8e390e1d2978&token=1388685340&lang=zh_CN#rd)

[部署系列——神经网络INT8量化教程第一讲！](https://zhuanlan.zhihu.com/p/405571578)

----
[NCNN Conv量化详解（一）](https://zhuanlan.zhihu.com/p/71881443)

[NCNN量化详解（二）](https://zhuanlan.zhihu.com/p/72375164)

[mmdeploy int8 量化 ncnn ViT part2](https://zhuanlan.zhihu.com/p/554022835)

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

### [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch)
- https://github.com/google-ai-edge/ai-edge-torch/issues/150
- https://github.com/google-ai-edge/ai-edge-torch/blob/main/ai_edge_torch/generative/quantize/quant_recipes.py

[Armv8 Neon technology](https://developer.arm.com/documentation/102474/0100/Fundamentals-of-Armv8-Neon-technology)

[cmake argument CPU architecture](https://github.com/Tencent/ncnn/blob/23890900c2a92a0932eba629d3c0bdbbc20808de/CMakeLists.txt#L267)