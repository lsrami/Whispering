#!/bin/bash


if ! command -v parallel &> /dev/null
then
    echo "没有安装parallel命令, 请使用apt/yum安装它!!!"
    echo "sudo apt install parallel"
    exit
fi

NJ=8 # 并行运行的数据集个数（进程数）
nj=16 # 单个数据集并行个数（线程数）
num_utts_per_shard=1000 # 单个tar包数据个数
data_type=raw # 数据类型
data_dir_list=data/dir.list # 数据集文件列表

export NJ
export nj
export num_utts_per_shard
export data_type
export data_dir_list

. tools/parse_options.sh || exit 1;

run_command() {

    data_dir=$1
    data_name=`basename $data_dir`
    if [ $data_type == "shard" ]; then
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads $nj $data_dir/wav.scp $data_dir/text --prefix $data_name \
        $(realpath $data_dir/shards) $data_dir/${data_name}_${data_type}_data.list >$data_dir/log.txt 2>&1 
    else
    tools/make_raw_list.py $data_dir/$x/wav.scp $data_dir/$x/text \
        $data_dir/${data_name}_${data_type}_data.list >$data_dir/log.txt 2>&1 
    fi
}

export -f run_command

echo "开始准备数据：$NJ 个进程 $nj 个线程 数据格式 $data_type"

# 在收到 SIGTERM 信号时，杀掉所有子进程
trap 'kill $(jobs -p)' SIGTERM
cat $data_dir_list| parallel -j$NJ run_command
