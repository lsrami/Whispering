#!/bin/bash

. ./path.sh || exit 1;

export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=DEBUG
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=0

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=1
num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

stage=1 
stop_stage=1

data_type=shard
train_config=conf/train_whisper.yaml
test_config="conf/test_whisper.yaml"
pretrain_model_dir=pretrain_model/openai/whisper-base # 指定预训练模型路径

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    for x in dev test train; do
          mkdir -p data/$x
    done
    echo "Please manually prepare the text and wav.scp files!!!"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Prepare data, prepare required format"
    num_utts_per_shard=1000

    for x in dev test train; do
      if [ $data_type == "shard" ]; then
        tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
          --num_threads 16 data/$x/wav.scp data/$x/text \
          $(realpath data/$x/shards) data/$x/data.list
      else
        tools/make_raw_list.py data/$x/wav.scp data/$x/text \
          data/$x/data.list
      fi
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # 多卡训练

    torchrun --rdzv_backend=c10d  --rdzv_id=whispering --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=${num_gpus} --max_restarts=3 \
          whispering/bin/train.py \
          --config $train_config \
          --data_type $data_type \
          --train_data data/train/data.list \
          --cv_data data/dev/data.list \
          --save_model_dir finetuned_model/whispering \
          --pretrain_model_dir $pretrain_model_dir \
          --metric_type cer
          # --resume_flag # 是否恢复训练
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # 测试
    python whispering/bin/test.py \
          --config $test_config \
          --data_type $data_type \
          --test_data data/test/data.list \
          --result_path finetuned_model/whispering/test_cer.txt \
          --pretrain_model_dir finetuned_model/whispering/checkpoint_best \
          --metric_type cer
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    # (备用)手动多卡训练
    gpu_list=${CUDA_VISIBLE_DEVICES}
    random_port=$((RANDOM % 32768 + 32768))
    num_gpus=$(echo "$gpu_list" | awk -F',' '{print NF}')
    num_nodes=1
    node_rank=0
    export WORLD_SIZE=$(expr $num_gpus \* $num_nodes)
    export MASTER_ADDR=localhost
    export MASTER_PORT=${random_port}

    train_config=conf/train_whisper.yaml
    data_type=shard
    for ((i = 0; i < ${num_gpus}; ++i)); do
    {
        gpu_id=$(echo $gpu_list | cut -d',' -f$[$i+1])
        export RANK=$(expr $node_rank \* $num_gpus + $i)
        export LOCAL_RANK=$i

        CUDA_VISIBLE_DEVICES=$gpu_id python whispering/bin/train.py \
            --config $train_config \
            --data_type $data_type \
            --train_data data/train/data.list \
            --cv_data data/dev/data.list \
            --save_model_dir finetuned_model/whispering \
          --pretrain_model_dir $pretrain_model_dir \
          --metric_type cer
          # --resume_flag
    } &
    done
    wait
fi




