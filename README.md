(简体中文|[English](./docs/README_en.md))

# Whispering: 动态多语言、多任务Whisper模型训练框架
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.8,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.13.0-blue"></a>
</p>

Whispering支持[OpenAI](https://huggingface.co/openai)在hugging face上开源的Whisper所有型号模型预训练和微调，使用[UIO](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md)方式进行数据加载，极大提升了大数据量训练IO瓶颈，该框架已在上万小时数据集上得到验证，训练稳定，快速高效。

<div align="center">  
<h4>
 <a href="#核心功能"> 核心功能 </a>   
｜<a href="#环境安装"> 环境安装 </a>
｜<a href="#模型下载"> 模型下载 </a>
｜<a href="#准备数据"> 准备数据 </a>
｜<a href="#快速开始"> 快速开始 </a>
｜<a href="#联系我们"> 联系我们 </a>
｜<a href="#致谢名单"> 致谢名单 </a>
</h4>
</div>

<a name="核心功能"></a>
## 核心功能
- 支持一次同时做语音识别、语音翻译、VAD等多个任务，多个语言训练
- 支持raw/shard两种训练数据格式
- 支持static/dynamic两种训练batch类型
- 支持spec_aug、shuffle增强等数据增强方式
- 支持cer、wer、bleu等多个指标选择最优模型


<a name="环境安装"></a>
## 环境安装
- 硬性要求：torch>=1.13.0 transformers>=4.28.0
```bash
conda create -n whispering python==3.10
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate whispering

pip install -r requirements.txt
```

<a name="模型下载"></a>
## 模型下载
请从[openai/whisper](https://huggingface.co/models?search=openai/whisper)下载预训练模型

```bash
mkdir pretrain_model/ && cd pretrain_model/

git clone https://huggingface.co/openai/whisper-base
```

注意：官方模型提供的`config.json`中`bos_token_id`和`eos_token_id`的值都为50257，这可能是一个bug

因此[做padding](./whispering/dataset/processor.py#L606)时使用了指向50258的`decoder_start_token_id`去除labels中的第一个token，而不是官方教程中的`bos_token_id`


<a name="准备数据"></a>
## 准备数据

首先准备好text和wav.scp文件，然后使用提供的脚本自动转换为raw或者shard训练数据格式

1. 创建train dev test文件夹
```bash
cd examples/aishell/s0

bash run.sh --stage -1 --stop_stage -1
```
2. 手动生成text和wav.scp文件放到train dev test文件夹下面
- 单语言单任务数据text和wav.scp样例
```bash
==> text <==
BAC009S0002W0122 而对楼市成交抑制作用最大的限购
BAC009S0002W0123 也成为地方政府的眼中钉
BAC009S0002W0124 自六月底呼和浩特市率先宣布取消限购后

==> wav.scp <==
BAC009S0002W0122 /data_aishell/wav/train/S0002/BAC009S0002W0122.wav
BAC009S0002W0123 /data_aishell/wav/train/S0002/BAC009S0002W0123.wav
BAC009S0002W0124 /data_aishell/wav/train/S0002/BAC009S0002W0124.wav
```
- 多语言多任务数据text和wav.scp样例
text中参数说明：

    所有参数均非必要，最小输入格式为`key {}`, 即无标注训练，等价于将sentence设置为`<|nospeech|>`

    `sentences`非必要（用于带时间戳训练），多个时间戳在`sentences`列表中增加
```bash
==> text <==
BAC009S0002W0122 {"key": "BAC009S0002W0122", "language": "chinese", "task": "transcribe", "sentence": "而对楼市成交抑制作用最大的限购", "sentences": [{"start": 0, "end": 6.0, "text": "而对楼市成交抑制作用最大的限购"}]}
BAC009S0002W0123 {"key": "BAC009S0002W0123", "language": "chinese", "task": "transcribe", "sentence": "也成为地方政府的眼中钉", "sentences": [{"start": 0, "end": 3.87, "text": "也成为地方政府的眼中钉"}]}
BAC009S0002W0124 {"key": "BAC009S0002W0124", "language": "chinese", "task": "transcribe", "sentence": "自六月底呼和浩特市率先宣布取消限购后", "sentences": [{"start": 0, "end": 5.41, "text": "自六月底呼和浩特市率先宣布取消限购后"}]}

==> wav.scp <==
BAC009S0002W0122 /data_aishell/wav/train/S0002/BAC009S0002W0122.wav
BAC009S0002W0123 /data_aishell/wav/train/S0002/BAC009S0002W0123.wav
BAC009S0002W0124 /data_aishell/wav/train/S0002/BAC009S0002W0124.wav
```

3. 生成训练数据格式data.list
```bash
# 确保 examples/aishell/s0/data 有如下文件
data/
├── dev
│   ├── text
│   └── wav.scp
├── test
│   ├── text
│   └── wav.scp
└── train
    ├── text
    └── wav.scp

# 生成raw/shard格式训练数据，大数据量推荐shard
bash run.sh --stage 0 --stop_stage 0 --data_type shard
```

<a name="快速开始"></a>
## 快速开始
训练阶段
```bash
bash run.sh --stage 1 --stop_stage 1
```
日志监控
```bash
# 查看训练日志
tail -f finetuned_model/whispering/train_log/log_2024-03-28_11-40-25.txt

# 查看tensorboard
tensorboard --host 0.0.0.0 --port 6006 --logdir finetuned_model/whispering/tensorboard/
```
测试阶段
```bash
bash run.sh --stage 2 --stop_stage 2

# 查看测试结果
tail finetuned_model/whispering/test_cer.txt
```

<a name="联系我们"></a>
## 联系我们
参考教程：
- [Aishell微调whisper-base](./docs/tutorial_aishell.md)
- [已知问题清单](./docs/issues_list.md)

如果您在使用中遇到其他问题，可以直接在github页面提Issues，欢迎语音兴趣爱好者进行交流和讨论。


<a name="致谢名单"></a>
## 致谢名单
1. dataloader及trainer大量参考[wenet](https://github.com/wenet-e2e/wenet)实现

2. tokenizer部分参考[Whisper-Finetune](https://github.com/yeyupiaoling/Whisper-Finetune)实现
