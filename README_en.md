(English|[Chinese](./README.md))

# Whispering: A dynamic multi-language, multi-task Whisper model training framework
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.8,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.13.0-blue"></a>
</p>

Whispering supports pre-training and fine-tuning of all Whisper models open-sourced by [OpenAI](https://huggingface.co/openai) on hugging face, greatly improving the IO bottleneck of large data volume training. This framework has been verified on datasets of tens of thousands of hours, with stable and efficient training.

<div align="center">  
<h4>
 <a href="#core-features"> Core Features </a>   
｜<a href="#environment-setup"> Environment Setup </a>
｜<a href="#model-download"> Model Download </a>
｜<a href="#data-preparation"> Data Preparation </a>
｜<a href="#quick-start"> Quick Start </a>
｜<a href="#contact-us"> Contact Us </a>
｜<a href="#acknowledgments"> Acknowledgments </a>
</h4>
</div>

<a name="core-features"></a>
## Core Features
- Supports multiple tasks such as speech recognition, speech translation, VAD, etc., in multiple languages simultaneously
- Supports raw/shard two types of training data formats
- Supports static/dynamic two types of training batch types
- Supports spec_aug, shuffle enhancement and other data enhancement methods
- Supports cer, wer, bleu and other indicators to select the optimal model


<a name="environment-setup"></a>
## Environment Setup
- Mandatory requirements: torch>=1.13.0 transformers>=4.28.0
```bash
conda create -n whispering python==3.10
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate whispering

pip install -r requirements.txt
```

<a name="model-download"></a>
## Model Download
Please download the pre-trained model from [openai/whisper](https://huggingface.co/models?search=openai/whisper)

```bash
mkdir pretrain_model/ && cd pretrain_model/

git clone https://huggingface.co/openai/whisper-base
```

Note: The `config.json` provided by the official model has both `bos_token_id` and `eos_token_id` values set to 50257, which might be a bug.

Therefore, when [doing padding](./whispering/dataset/processor.py#L603), the `decoder_start_token_id` pointing to 50258 is used to remove the first token in labels, instead of the `bos_token_id` in the official tutorial.

<a name="data-preparation"></a>
## Data Preparation

First, prepare the text and wav.scp files, then use the provided script to automatically convert to raw or shard training data format

1. Create train dev test folders
```bash
cd examples/aishell/s0

bash run.sh --stage -1 --stop_stage -1
```
2. Manually generate text and wav.scp files and place them under the train dev test folders
- Single language single task data text and wav.scp example
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
- Multi-language multi-task data text and wav.scp example
Explanation of parameters in the text:

    All parameters are not necessary, the minimum input format is key {}, i.e., training without annotation, equivalent to setting sentence to <|nospeech|>

    Sentences are not necessary (for training with timestamps), multiple timestamps can be added to the sentences list
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

3. Generate training data format data.list
```bash
# Make sure examples/aishell/s0/data has the following files
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

# Generate raw/shard format training data, shard is recommended for large data volume
bash run.sh --stage 0 --stop_stage 0 --data_type shard
```

<a name="quick-start"></a>
## Quick Start
Training phase
```bash
bash run.sh --stage 1 --stop_stage 1
```
Log monitoring
```bash
# View training log
tail -f finetuned_model/whispering/train_log/log_2024-03-28_11-40-25.txt

# View tensorboard
tensorboard --host 0.0.0.0 --port 6006 --logdir finetuned_model/whispering/tensorboard/
```
Testing phase
```bash
bash run.sh --stage 2 --stop_stage 2

# View test results
tail finetuned_model/whispering/test_cer.txt
```

<a name="contact-us"></a>
## Contact Us

If you encounter problems in use, you can directly raise Issues on the github page. We welcome enthusiasts of voice to communicate and discuss.


<a name="acknowledgments"></a>
## Acknowledgments
1. The dataloader and trainer largely refer to the implementation of [wenet](https://github.com/wenet-e2e/wenet)

2. The tokenizer part refers to the implementation of [Whisper-Finetune](https://github.com/yeyupiaoling/Whisper-Finetune)