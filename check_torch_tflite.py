import pickle
import os
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

root_dir_tensor = "dummy_input_pickles"
list_tensor_path = sorted(
    [
        os.path.join(root_dir_tensor, t) for t in os.listdir(root_dir_tensor) if
        not os.path.isdir(os.path.join(root_dir_tensor, t))
    ]
)
for tensor_path in list_tensor_path:
    with open(tensor_path, "rb") as f:
        data = pickle.load(f)
    print(type(data))
    # Print the loaded tensor
    print(data.shape)
    print(data.dtype)
    print("------")

root_dir_tflite = 'models_tflite/outputs_tflite_int8'
tflite_models = sorted([os.path.join(root_dir_tflite, lite) for lite in os.listdir(root_dir_tflite)])

for tflite_model_path in tqdm(tflite_models):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    input_details = interpreter.get_input_details()
    print(tflite_model_path)
    print(input_details[0])
    print(input_details[0]['shape'])  # 这里看看是不是 [1, 3, 224, 224] 固定死了
    print(input_details[0]['shape_signature'])  # 这里如果是 [-1, 3, -1, -1] 说明是动态的
    print("######")
