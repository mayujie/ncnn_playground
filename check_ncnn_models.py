import ncnn
import ncnn.model_zoo as model_zoo

ncnn_models = model_zoo.get_model_list()
for model_nc in ncnn_models:
    print(model_nc)