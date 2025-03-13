import os
import time
import tensorflow as tf
from tqdm import tqdm
from pydbp.client import ApiClient
from pydbp.update import update_loop

# get model from NAS
ROOT_DIR = "/nas/people/yujie_ma/ncnn_test/tflite_models/"
# list_models = sorted(
#     [
#         os.path.join(ROOT_DIR, lite) for lite in os.listdir(ROOT_DIR) if lite.endswith(".tflite")
#     ]
# )

list_models = [
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_efficientnet_b0_224.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_efficientnet_b0_256.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_efficientnet_b0_384.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_efficientnet_b0_512.tflite",
    #
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_mobilenet_v2_224.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_mobilenet_v2_256.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_mobilenet_v2_384.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_mobilenet_v2_512.tflite",
    #
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_resnet18_224.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_resnet18_256.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_resnet18_384.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_resnet18_512.tflite",
    #
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_resnet50_224.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_resnet50_256.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_resnet50_384.tflite",
    # "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_resnet50_512.tflite",

    "/nas/people/yujie_ma/ncnn_test/outputs_tflite_int8/torch_vit_b_16_224.tflite",

    # "/nas/people/yujie_ma/ncnn_test/tflite_models/torch_vit_b_16_224.tflite",
]

list_config_dict = [
    # {
    #     # GPU
    #     "thds": -1,
    #     "use_xnnpack": False,
    #     "use_gpu": True,
    #     "gpu_backend": "cl",
    # },
    {
        # Thread 8
        "thds": 8,
        "use_xnnpack": True,
        "use_gpu": False,
        "gpu_backend": None,
    },
    {
        # Thread 4
        "thds": 4,
        "use_xnnpack": True,
        "use_gpu": False,
        "gpu_backend": None,
    },
    {
        # Thread 2
        "thds": 2,
        "use_xnnpack": True,
        "use_gpu": False,
        "gpu_backend": None,
    },
    {
        # Thread 1
        "thds": 1,
        "use_xnnpack": True,
        "use_gpu": False,
        "gpu_backend": None,
    },
]

list_input = [
    "/nas/people/yujie_ma/ncnn_test/in_tensor_224.pkl",
    "/nas/people/yujie_ma/ncnn_test/in_tensor_256.pkl",
    "/nas/people/yujie_ma/ncnn_test/in_tensor_384.pkl",
    "/nas/people/yujie_ma/ncnn_test/in_tensor_512.pkl",
]

# api client
api_client = ApiClient(
    # api_url=None,
    # token=None,
    api_url="http://10.70.227.35/api/",
    token="e9a4492c8ba7826ebab3f22bdf4b12b4fe3d6e00"
)
project = api_client.get_project(project_id=61)

manager = api_client.get_tf_manager(project)

executable = manager.get_executable(release_version="16.01.2025")

save_dir = "results_tflite_benchmarks"
LOG_FILENAME = f"benchmark_results_0312_tflite_int8_vit.log"
log_file = os.path.join(save_dir, LOG_FILENAME)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Open log file in append mode
with (open(log_file, "a") as f):
    for model in tqdm(list_models, desc="benchmark model"):
        # if "vit_b_16" in model:
        #     print(f'skip {model}......')
        #     continue

        interpreter = tf.lite.Interpreter(model_path=model)
        input_details = interpreter.get_input_details()

        for config_dict in tqdm(list_config_dict, desc="benchmark config"):
            # for infer_input in tqdm(list_input, desc="infer input"):
            bench_comment = f"{os.path.basename(model)} " + f" |{input_details[0]['shape']} {input_details[0]['dtype']}| " + " | ".join(
                f"{key}={value}" for key, value in config_dict.items()
            )
            # ) + f" {os.path.basename(infer_input)}"

            # run benchmark
            benchmark = manager.run_benchmark(
                model,
                executable,
                num_runs=500,
                # num_runs=32,
                max_secs=150,
                num_threads=config_dict["thds"],
                enable_op_profiling=False,
                use_xnnpack=config_dict["use_xnnpack"],
                use_nnapi=False,
                use_hexagon=False,
                use_gpu=config_dict["use_gpu"],
                gpu_backend=config_dict["gpu_backend"],
                gpu_fp16=False,
                require_full_delegation=False,
                comment=bench_comment,
                # inference_input=infer_input,
                inference_input=None,
                cli_extra_args=None,
                # cli_extra_args="--print_preinvoke_state=true",
            )

            # wait for benchmark output, you can do it on your own
            # or use default checking loop, enable verbose to see
            # some underhood communication
            update_loop(benchmark, verbose=False)

            # Get benchmark results
            benchmark_output = benchmark.get_benchmark_output()
            latency_us = benchmark.latency_avg
            latency_ms = float(latency_us) / 1000

            # Format log entry
            log_entry = (
                f"{bench_comment}\n"
                f"### Inference (avg): {latency_us} us\n"
                f"### Inference (avg): {latency_ms} ms\n"
                "------------------------------------------\n"
            )
            # Print to console
            # print(benchmark.api_response)
            print(log_entry)

            # Write to log file
            f.write(log_entry)

            # Wait before next run
            time.sleep(8)
