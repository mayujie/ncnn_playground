#!/system/bin/sh

# Extract the filename without the path
PARAM_FILE=$(basename "$6")   # Extract just the filename from the PARAM path
PARAM_NAME="${PARAM_FILE%.*}" # Remove the file extension (e.g., .param)

# Define the log file name dynamically
LOG_FILE="result_benchmark_${PARAM_NAME}.log"

exec >"$LOG_FILE" 2>&1   # Redirect output to log file
tee -a "$LOG_FILE" <&0 & # Append log file to console

NUM_LOOP=${1:-64}                     # Use first argument or default to 64
NUM_THREADS=${2:-1}                   # Use second argument or default to 1
POWER_SAVE=${3:-0}                    # Use third argument or default to 0
GPU_DEVICE=${4:-0}                    # Use fourth argument or default to 0
COOLING_DOWN=${5:-0}                  # Use fifth argument or default to 0
PARAM="${6:-EfficientNet.ncnn.param}" # Use sixth argument or default to ${PARAM}

# Store input shapes in an array
INPUT_SHAPES=("[512,512,3,1]" "[384,384,3,1]" "[256,256,3,1]" "[224,224,3,1]")

echo "========== Benchmark Run at $(date) ==========" >>"$LOG_FILE"
echo "Running with NUM_LOOP=${NUM_LOOP}, NUM_THREADS=${NUM_THREADS}, GPU_DEVICE=${GPU_DEVICE}"
echo "Logging results to ${LOG_FILE}"
echo "---------------------------------------"

# Loop through each input shape
for IN_SHAPE in "${INPUT_SHAPES[@]}"; do
  echo "Running benchmark for ${PARAM} with shape ${IN_SHAPE}..."
  ./benchncnn ${NUM_LOOP} ${NUM_THREADS} ${POWER_SAVE} ${GPU_DEVICE} ${COOLING_DOWN} param=${PARAM} shape="${IN_SHAPE}"
  echo "DONE for ${PARAM} with ${IN_SHAPE}"
  echo
done

echo "Benchmark completed at $(date)"
echo "======================================="
