#!/bin/bash

# Define the save directory
save_dir="results_ncnn_fp16"

# Create the save directory if it doesn't exist
mkdir -p "$save_dir"

# Get the list of files from the device
files=$(adb shell ls /data/local/tmp/result_benchmark_*.ncnn.log)

# Check if there are matching files
if [ -z "$files" ]; then
    echo "No matching log files found on the device."
    exit 1
fi

# Pull each file individually
for file in $files; do
    echo "Pulling $file..."
    adb pull "$file" "$save_dir/"
done

echo "All files have been pulled to $save_dir."
