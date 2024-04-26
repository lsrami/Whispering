#!/bin/bash

. ./path.sh || exit 1;

export NCCL_DEBUG=INFO # 日志等级
export NCCL_SOCKET_IFNAME=eno1 # socket网卡名称，需要改为实际网卡
export NCCL_P2P_DISABLE=0 # 使用NVlink或者IB时，进行gpu直接通信，默认0
export NCCL_IB_DISABLE=0 # 是否禁用IB传输，默认0
export NCCL_IB_HCA=mlx5 # IB通信时必须设置的IB网卡名，mlx5_0:1 使用卡mlx5_0的端口 1
export NCCL_SHM_DISABLE=0 # 是否禁用共享内存传输，默认0

# export NCCL_NET_GDR_READ=1 # 直接从GPU内存读取绕过cpu内存，默认0
# export NCCL_P2P_LEVEL=1 # 建议不要设置，默认自动选择

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=1
num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

stage=1
stop_stage=1

data_type=shard # raw/shard
train_config=conf/train_whisper.yaml
test_config=conf/test_whisper.yaml
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
    torchrun --rdzv_backend=c10d  --rdzv_id=whispering --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=${num_gpus} \
          whispering/bin/train.py \
          --config $train_config \
          --data_type $data_type \
          --train_data data/train/data.list \
          --cv_data data/dev/data.list \
          --save_model_dir finetuned_model/whispering \
          --pretrain_model_dir $pretrain_model_dir \
          --metric_type cer \
          --task transcribe \
          --language chinese \
          --use_smooth_loss

    # 其他训练参数说明
    # --resume_train # 是否从上次中断的地方继续训练，必须把 --pretrain_model_dir 指为上次中断的模型，而不是初始预训练模型
    # --label_json # 文本标签的类型是否为json类型，在动态多语言、多任务训练时必须指定
    # --timestamps # 是否使用带时间戳数据训练，只在指定 --label_json 后才生效
    # --cv_data_type # 单独为cv dataset指定data_type
    # --max_keep_checkpoint # 最多保留 N 个checkpoint
    # --use_smooth_loss # 是否使用标签平滑损失
    # --cv_partition # 训练过程中是否多卡验证
    # --monitor_train # 如果设置则 监控训练进程通信
    # --monitor_cv # 如果设置则 监控验证进程通信
    # --timeout # 进程通信超时时间
    # --one_tar_nums # 一个tar包中音频个数，用于计算训练进度
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # 测试
    python whispering/bin/test.py \
          --config $test_config \
          --data_type $data_type \
          --test_data data/test/data.list \
          --result_path finetuned_model/whispering/test_cer.txt \
          --pretrain_model_dir finetuned_model/whispering/checkpoint_best \
          --metric_type cer \
          --task transcribe \
          --language chinese

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




