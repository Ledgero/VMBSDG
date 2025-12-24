#!/usr/bin/env bash
set -euo pipefail

TRAIN_IMG_SIZE=640
data_cfg_path="configs/data/${TRAIN_IMG_SIZE}p/megadepth_trainval_${TRAIN_IMG_SIZE}-360.py"
# data_cfg_path="configs/data/832p/megadepth_trainval_832-200.py"
main_cfg_path="configs/jamma/outdoor/final.py"

pin_memory=true
exp_name="roma-360_${TRAIN_IMG_SIZE}_aug"
mkdir -p roma_log/${TRAIN_IMG_SIZE}/${exp_name}

############################
n_gpus_per_node=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
batch_size=3

export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1200
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

choose_port() {
    for port in $(seq 29500 29519); do
        if ! lsof -i:$port &>/dev/null; then
            echo $port
            return
        fi
    done
    echo "没有找到可用端口 29500-29519" >&2
    exit 1
}
MASTER_ADDR=127.0.0.1
MASTER_PORT=$(choose_port)

echo "[INFO] Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"


torchrun --nproc_per_node=${n_gpus_per_node} --nnodes=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    experiments/train_roma_outdoor.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --batch_size=${batch_size} \
    --loftr \
    2> >(tee /dev/stderr | grep -v "Epoch" >>roma_log/${TRAIN_IMG_SIZE}/${exp_name}/debug.log)