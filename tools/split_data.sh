#!/bin/bash

NJ=8 # 切分成NJ份
scp=wav.scp
txt=text
save_dir=train_split # 切分后的保存目录

. tools/parse_options.sh || exit 1;

# 创建存放分片的文件夹
mkdir -p $save_dir

sort -u -k1 $scp >$save_dir/wav_sort.scp
sort -u -k1 $txt >$save_dir/text_sort

scp_lines=`wc -l < $save_dir/wav_sort.scp | awk '{print $1}'`
txt_lines=`wc -l < $save_dir/text_sort | awk '{print $1}'`

scp_split_lines=$(( scp_lines / NJ + 1 ))
txt_split_lines=$(( txt_lines / NJ + 1 ))

echo "音频行数：$scp_lines 文本行数: $txt_lines 分片个数: $NJ 音频分片行数: $scp_split_lines 文本分片行数: $txt_split_lines"
# 将文本文件分割成NJ个部分
split -d -a 4 -l  $scp_split_lines $save_dir/wav_sort.scp $save_dir/wav_
split -d -a 4 -l  $txt_split_lines $save_dir/text_sort $save_dir/text_

# 为每个子文件夹创建wav.scp和text文件
> $save_dir/sub_dir.list
for i in $(seq 0 $((NJ-1))); do
    sub_dir=$save_dir/split_$i

    mkdir $sub_dir
    realpath $sub_dir >> $save_dir/sub_dir.list
    mv $save_dir/wav_$(printf "%04d" $i) $sub_dir/wav.scp
    mv $save_dir/text_$(printf "%04d" $i) $sub_dir/text
done

echo -e "文件切片列表： $save_dir/sub_dir.list \ndone!"
