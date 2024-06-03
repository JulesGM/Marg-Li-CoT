#!/usr/bin/env bash
set -u
set -e


# Changes
VAL_SUBSET_SIZE=100

# Never changes
MIXED_PRECISION=no
TOTAL_PROCESSES=$(("$SLURM_NNODES" * "$SLURM_GPUS_ON_NODE"))
SERVER_PORT=25905
SERVER_HOSTNAME="$HOSTNAME"
NUM_NODES="$SLURM_NNODES"
DEBUG=false

###############################################################################
# Rest is fixed
###############################################################################
BRED='\033[1;31m'
BGREEN='\033[1;32m'
BBLUE='\033[1;34m'
BYELLOW='\033[1;33m'
NC='\033[0m' # No Color

ACCELERATE_BIN_MULTI_NODE=("srun" "accelerate")
ACCELERATE_BIN_SINGLE_NODE=("accelerate")
CONDITIONAL_ARGS_MULTI_NODE=("--deepspeed_multinode_launcher=standard")
CONDITIONAL_ARGS_SINGLE_NODE=("")

if [[ "${NUM_NODES}" > 1 ]] ; then
    ACCELERATE_BIN="${ACCELERATE_BIN_MULTI_NODE[@]}"
    CONDITIONAL_ARGS="${CONDITIONAL_ARGS_MUTLI_NODE[@]}"
else
    ACCELERATE_BIN="${ACCELERATE_BIN_SINGLE_NODE[@]}"
    CONDITIONAL_ARGS="${CONDITIONAL_ARGS_SINGLE_NODE[@]}"
fi

# The last argument needs to be the path of the bin
if $DEBUG; then
    BIN_PATH=(--no_python "$PWD/inner_launch.sh")
else
    BIN_PATH=("$PWD/bin_exp.py")
fi

# raise if BIN_PATH does not exist
if [[ ! -f "${BIN_PATH[-1]}" ]]; then
   echo -e "${BRED}Expected bin path to exist: ${BIN_PATH[-1]}${NC}"
   exit 1
fi

# If --one is passed, use only one process total.
if [[ "$#" -gt 0 ]]; then
    if [[ "$1" != "--one" ]]; then
        echo -e "${BRED}Expected \$1 to be \"one\", is: \"$1\"${NC}"
        exit 1
    fi

    echo -e "${BYELLOW}Only using one process, to debug.${NC}"
    TOTAL_PROCESSES=1
    NUM_NODES=1
    ACCELERATE_BIN="${ACCELERATE_BIN_SINGLE_NODE[@]}"
    SERVER_HOSTNAME=""
    CONDITIONAL_ARGS="${CONDITIONAL_ARGS_SINGLE_NODE[@]}"
fi

# Extract the GPU models
GPU_MODELS="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq)"
IFS=$'\n' read -ra GPU_MODELS <<< "$GPU_MODELS"

# If any of the GPUs is not a A100, disable bf16
for MODEL in "${GPU_MODELS[@]}"; do
    if [[ "${MODEL}" != *"A100"* ]]; then
        echo -e "${BBLUE}Disabling bf16 because of GPU model: $MODEL"
        MIXED_PRECISION=no
        break
    fi
done

if [[ "$VAL_SUBSET_SIZE" != "None" ]] ; then
    echo -e "${BRED}>>> Using a subset! Of size: ${VAL_SUBSET_SIZE}${NC}"
fi

# Kill all wandb servers
pgrep wandb | xargs kill -9 2>/dev/null || true

# Kill all python processes
killall -9 python 2>/dev/null || true

# Make it so the accelerate launch command is printed
set -x

# Run the training
"${ACCELERATE_BIN[@]}" launch               \
    --main_process_port="${SERVER_PORT}"    \
    --main_process_ip="${SERVER_HOSTNAME}"  \
    --num_processes="${TOTAL_PROCESSES}"    \
    --num_machines="${NUM_NODES}"           \
    --mixed_precision="${MIXED_PRECISION}"  \
    "${CONDITIONAL_ARGS[@]}"                \
    "${BIN_PATH[@]}"                        \
    --val_subset_size "${VAL_SUBSET_SIZE}"
