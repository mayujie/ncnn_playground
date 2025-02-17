import os
import time
from tqdm import tqdm
from pydbp.client import ApiClient
from pydbp.update import update_loop

ROOT_DIR = "/nas/people/yujie_ma/ncnn_test/tflite_models/"
# get model from NAS
list_models = [
    "torch_efficientnet_b0.tflite",
    # "torch_mobilenet_v2.tflite",
    # "torch_resnet18.tflite",
    # "torch_resnet50.tflite",
    # "torch_vit_b_16.tflite",
]

list_models = [(os.path.join(ROOT_DIR, i)) for i in list_models]

list_config_dict = [
    {
        # GPU
        "thds": -1,
        "use_xnnpack": False,
        "use_gpu": True,
        "gpu_backend": "cl",
    },
    # {
    #     # Thread 8
    #     "thds": 8,
    #     "use_xnnpack": True,
    #     "use_gpu": False,
    #     "gpu_backend": None,
    # },
    # {
    #     # Thread 4
    #     "thds": 4,
    #     "use_xnnpack": True,
    #     "use_gpu": False,
    #     "gpu_backend": None,
    # },
    # {
    #     # Thread 2
    #     "thds": 2,
    #     "use_xnnpack": True,
    #     "use_gpu": False,
    #     "gpu_backend": None,
    # },
    # {
    #     # Thread 1
    #     "thds": 1,
    #     "use_xnnpack": True,
    #     "use_gpu": False,
    #     "gpu_backend": None,
    # },
]

list_input = [
    "/nas/people/yujie_ma/ncnn_test/in_tensor_224.pkl",
    "/nas/people/yujie_ma/ncnn_test/in_tensor_256.pkl",
    "/nas/people/yujie_ma/ncnn_test/in_tensor_384.pkl",
    "/nas/people/yujie_ma/ncnn_test/in_tensor_512.pkl",
]

# api client
api_client = ApiClient(
    api_url="http://10.70.227.35/api/",
    token="e9a4492c8ba7826ebab3f22bdf4b12b4fe3d6e00"
)
project = api_client.get_project(project_id=56)

manager = api_client.get_tf_manager(project)

executable = manager.get_executable(release_version="16.01.2025")

save_dir = "outputs_tflite_benchmarks"
log_file = os.path.join(save_dir, "benchmark_results.log")
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Open log file in append mode
with open(log_file, "a") as f:
    for model in tqdm(list_models, desc="benchmark model"):
        for config_dict in tqdm(list_config_dict, desc="benchmark config"):
            for infer_input in tqdm(list_input, desc="infer input"):
                bench_comment = f"{os.path.basename(model)}" + " " + " | ".join(
                    f"{key}={value}" for key, value in config_dict.items()
                ) + f" {os.path.basename(infer_input)}"
                # run benchmark
                benchmark = manager.run_benchmark(
                    model,
                    executable,
                    num_runs=500,
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
                    inference_input=infer_input,
                )

                # wait for benchmark output, you can do it on your own
                # or use default checking loop, enable verbose to see
                # some underhood communication
                update_loop(benchmark, verbose=False)

                # Get benchmark results
                benchmark_output = benchmark.get_benchmark_output()
                latency_us = benchmark.latency_avg
                latency_ms = latency_us / 1000

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
