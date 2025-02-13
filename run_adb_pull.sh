#!/bin/sh
OUTPUT_DIR="${1:-results_ncnn_bench_thd_4}"
mkdir $OUTPUT_DIR

for file in $(adb shell ls "/data/local/tmp/result_benchmark*.log" | tr -d '\r'); do
  adb pull "${file}" "${OUTPUT_DIR}"
done
