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

import argparse
import os
import time
import json

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from whispering.dataset.dataset import Dataset
from whispering.utils.checkpoint import load_checkpoint
from whispering.utils.file_utils import setup_logger
from whispering.metrics import load_metric


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--test_data', required=True, help='cv data file')
    parser.add_argument('--result_path', required=True, help='save model dir')
    parser.add_argument('--pretrain_model_dir',
                        required=True, help='pretrain model dir')
    parser.add_argument('--task',
                        default='transcribe',
                        type=str,
                        choices=['transcribe', 'translate'],
                        help='Training task type')
    parser.add_argument('--language',
                        default='chinese',
                        type=str,
                        help='language')
    parser.add_argument('--metric_type',
                        default='cer',
                        type=str,
                        choices=['cer', 'wer', 'bleu'],
                        help='metric type. Choose from: cer, wer, bleu')
    parser.add_argument('--label_json',
                        action='store_true',
                        default=False,
                        help='Whether to use json format labels')
    parser.add_argument('--timestamps',
                        action='store_true',
                        default=False,
                        help='timestamps')

    args = parser.parse_args()
    return args


def main(args):
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    dataset_conf = configs['dataset_conf']
    dataloader_conf = configs.get('dataloader_conf', {})
    metric_type = args.metric_type
    metric = load_metric(metric_type)

    assert not dataset_conf['speed_perturb'], f"speed_perturb spec_aug spec_sub 必须为 False"

    model, whisper_processor, _, _, _ = load_checkpoint(
        args.pretrain_model_dir,
        args.language,
        args.task,
        args.timestamps,
        device=device)

    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps)

    test_dataset, test_one_card_utts = Dataset(args.data_type,
                                               args.test_data,
                                               dataset_conf,
                                               args.label_json,
                                               args.timestamps,
                                               partition=False,
                                               whisper_processor=whisper_processor)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=dataloader_conf.get(
                                      'num_workers', 4),
                                  pin_memory=dataloader_conf.get(
                                      'pin_memory', True),
                                  prefetch_factor=dataloader_conf.get(
                                      'prefetch_factor', 8),
                                  drop_last=dataloader_conf.get(
                                      'drop_last', False),
                                  )

    seen_num_utts, total_scores = 0, 0
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_data_loader), "running", total=test_one_card_utts):
                keys, feats, labels = batch
                feats = feats.to(device)
                labels = labels.to(device)
                labels = torch.where(labels != -100, labels,
                                     whisper_processor.tokenizer.pad_token_id)

                generated_preds = model.generate(inputs=feats,
                                                 decoder_input_ids=labels[:, :3],
                                                 max_new_tokens=443,
                                                 return_timestamps=args.timestamps)

                decoded_labels = whisper_processor.batch_decode(
                    labels, skip_special_tokens=True)
                decoded_preds = whisper_processor.batch_decode(
                    generated_preds, skip_special_tokens=True)

                # todo:添加必要的文本后处理
                # 分词、ITN、去标点、去时间戳、繁转简

                for key, reference, prediction in zip(keys, decoded_labels, decoded_preds):
                    scores = metric.compute(
                        references=[reference], predictions=[prediction])
                    if args.metric_type == 'bleu':
                        scores = scores["bleu"]
                    seen_num_utts += 1
                    total_scores += scores

                    result = {'key': key, 'reference': reference,
                              'prediction': prediction, metric_type: round(scores, 4)}
                    logger.info(json.dumps(result, ensure_ascii=False))

    if seen_num_utts > 0:
        avg_scores = total_scores / seen_num_utts
        logger.info(f"\n{'='*10}Test Completed{'='*10}\n")
        logger.info(
            f"seen_num_utts: {seen_num_utts} metric_type: {metric_type} avg_scores: {round(avg_scores, 4)}")
    else:
        logger.error(f"error: seen_num_utts is zero!")


if __name__ == '__main__':
    args = get_args()
    log_path = args.result_path

    os.makedirs(os.path.split(log_path)[0], exist_ok=True)

    logger = setup_logger('test_logger', log_path,
                          level='DEBUG', fmt='%(message)s')
    main(args)
