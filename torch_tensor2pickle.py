import os
import torch
import pickle


list_shape = [512, 384, 256, 224]

for item_shape in list_shape:
    x = torch.randn(1, 3, item_shape, item_shape)

    save_dir = "dummy_input_pickles"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Save to a pickle file
    with open(f"{save_dir}/in_tensor_{item_shape}.pkl", "wb") as f:
        pickle.dump(x, f)