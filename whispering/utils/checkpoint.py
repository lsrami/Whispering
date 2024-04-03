# Copyright (c) 2024 Timekettle Inc. (authors: Sirui Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shutil
import glob

import yaml
import torch

import datetime
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def _load_processor(model_dir, language, task, no_timestamps) -> WhisperProcessor:
    whisper_processor = WhisperProcessor.from_pretrained(model_dir,
                                                         language=language,
                                                         task=task,
                                                         no_timestamps=no_timestamps,
                                                         local_files_only=True)
    return whisper_processor


def _load_model(model_dir, device='auto'):
    model = WhisperForConditionalGeneration.from_pretrained(model_dir,
                                                            load_in_8bit=False,
                                                            device_map=device,
                                                            local_files_only=True)
    configs = {}
    info_path = os.path.join(model_dir, 'train_infos.yaml')
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

    optimizer_state_dict, scheduler_state_dict = None, None
    optimizer_path = os.path.join(model_dir, 'optimizer.pth')
    if os.path.exists(optimizer_path):
        optimizer_state_dict = torch.load(optimizer_path, map_location='cpu')
    scheduler_path = os.path.join(model_dir, 'scheduler.pth')
    if os.path.exists(scheduler_path):
        scheduler_state_dict = torch.load(scheduler_path, map_location='cpu')
    return model, configs, optimizer_state_dict, scheduler_state_dict


def _save_processor(whisper_processor: WhisperProcessor, model_dir: str):
    # 为旧版模型添加时间戳token
    vocab = whisper_processor.tokenizer.get_vocab()
    timestamp_begin = vocab.get('<|0.00|>')
    if timestamp_begin is None:
        timestamps_token = [f"<|{i / 100.0:.2f}|>" for i in range(0, 3001, 2)]
        whisper_processor.tokenizer.add_tokens(timestamps_token)

    whisper_processor.save_pretrained(model_dir)


def _save_model(model: WhisperForConditionalGeneration, model_dir: str, optimizer, scheduler, infos):

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.save_pretrained(model_dir)
    info_path = os.path.join(model_dir, 'train_infos.yaml')
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)

    optimizer_path = os.path.join(model_dir, 'optimizer.pth')
    torch.save(optimizer.state_dict(), optimizer_path)

    scheduler_path = os.path.join(model_dir, 'scheduler.pth')
    torch.save(scheduler.state_dict(), scheduler_path)


def load_checkpoint(model_dir, language, task, timestamps, device='auto'):
    no_timestamps = not timestamps
    whisper_processor = _load_processor(
        model_dir, language, task, no_timestamps)
    model, configs, optimizer_state_dict, scheduler_state_dict = _load_model(
        model_dir, device=device)

    return model, whisper_processor, configs, optimizer_state_dict, scheduler_state_dict


def save_checkpoint(model, whisper_processor, model_dir, optimizer, scheduler, infos=None):
    if infos is None:
        infos = {}

    root_model_dir, _ = os.path.split(model_dir)
    model_files = glob.glob(os.path.join(root_model_dir, "checkpoint_epoch*"))

    model_files.sort(key=os.path.getmtime)
    max_keep_checkpoint = infos.get('max_keep_checkpoint', 5)
    assert max_keep_checkpoint > 0
    while len(model_files) > max_keep_checkpoint - 1:
        shutil.rmtree(model_files.pop(0), ignore_errors=True)

    _save_processor(whisper_processor, model_dir)
    _save_model(model, model_dir, optimizer, scheduler, infos)



def save_checkpoint_best(current_metric, best_metric, model_dir, best_model_dir, maximize=False):
    """
    current_metric: Evaluation metric for the current model
    best_metric: Evaluation metric for the best model so far
    model_dir: Directory where the current model is saved
    best_model_dir: Directory where the best model is saved
    maximize: If True, then a higher evaluation metric is better; if False, a lower evaluation metric is better
    """
    is_best = best_metric < current_metric if maximize else best_metric > current_metric

    if is_best:
        dir_name, base_name = os.path.split(best_model_dir)
        tmp_dir = os.path.join(dir_name, base_name + '_old')

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(best_model_dir):
            os.rename(best_model_dir, tmp_dir)
        shutil.copytree(model_dir, best_model_dir)
        if os.path.exists(best_model_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return is_best
