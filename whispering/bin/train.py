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

from __future__ import print_function

import argparse
import copy
import os
import time

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from datetime import timedelta
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from whispering.dataset.dataset import Dataset
from whispering.utils.checkpoint import load_checkpoint, save_checkpoint
from whispering.utils.executor import Executor
from whispering.utils.scheduler import WarmupLR, NoamHoldAnnealing
from whispering.utils.config import override_config
from whispering.utils.file_utils import setup_logger


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--cv_data_type',
                        default=None,
                        choices=['raw', 'shard'],
                        help='cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--save_model_dir', required=True,
                        help='save model dir')
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
                        default='loss',
                        type=str,
                        choices=['loss', 'cer', 'wer', 'bleu'],
                        help='best model preservation metrics')
    parser.add_argument('--max_keep_checkpoint',
                        default=5,
                        type=int,
                        help='max keep checkpoint')
    parser.add_argument('--label_json',
                        action='store_true',
                        default=False,
                        help='Whether to use json format labels')
    parser.add_argument('--timestamps',
                        action='store_true',
                        default=False,
                        help='Whether to use timestamp training')
    parser.add_argument('--timeout',
                        default=180,
                        type=int,
                        help='Dist timeout (in minutes)')
    parser.add_argument('--one_tar_nums',
                        default=1000,
                        type=int,
                        help='Data entries number in a tar')
    parser.add_argument('--resume_train',
                        action='store_true',
                        default=False,
                        help='Resume training from checkpoint')
    parser.add_argument('--use_smooth_loss',
                        action='store_true',
                        default=False,
                        help='Use label smoothing loss')
    parser.add_argument('--cv_partition',
                        action='store_true',
                        default=False,
                        help='Use multiple cards for cross-validation. '
                        'Note: When cv_data_type is "shard" and the number of audio '
                        'tracks in a tar package exceeds 500, there is a chance '
                        'that the process will be interrupted. Solutions: '
                        '1. Do not set --cv_partition '
                        '2. Set --cv_data_type=raw '
                        '3. Reduce the number of audio tracks in the tar')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    args = parser.parse_args()
    return args


def main(args):
    # Set random seed
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # dist configs
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ.get('LOCAL_RANK'))
        torch.cuda.set_device(local_rank)  # 如果使用手动多进程，请注释
        dist.init_process_group(
            backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=timedelta(minutes=args.timeout))

    # Load model and tokenizer
    model, whisper_processor, infos, optimizer_state_dict, scheduler_state_dict = load_checkpoint(
        args.pretrain_model_dir,
        args.language,
        args.task,
        args.timestamps,
        device=device)
    # Randomly initialize the model
    if False:
        model.init_weights()
    # Parameters for dataset and dataloader
    train_dataset_conf = configs['dataset_conf']
    cv_dataset_conf = copy.deepcopy(train_dataset_conf)
    cv_dataset_conf['speed_perturb'] = False
    cv_dataset_conf['spec_aug'] = False
    cv_dataset_conf['spec_sub'] = False
    cv_dataset_conf['shuffle'] = False
    dataloader_conf = configs.get('dataloader_conf', {})
    # backward compatibility
    cv_dataloader_conf = configs.get('cv_dataloader_conf', dataloader_conf)
    cv_partition = args.cv_partition
    if args.cv_data_type is None:
        args.cv_data_type = args.data_type

    # Retrieve parameters from the point of last interruption
    if args.resume_train:
        step = infos.get('step', -1) + 1
        start_epoch = infos.get('epoch', 0)
        start_batch = infos.get('batch_idx', -1) + 1
        metric_type = infos.get('metric_type', 'loss')
        best_metric = infos.get('best_metric', float('inf'))
    else:
        metric_type = args.metric_type
        step, start_epoch, start_batch = 0, 0, 0
        best_metric = float('-inf') if metric_type == 'bleu' else float('inf')

    # Get the training parameters
    train_conf = configs.get('train_conf', {})
    max_step = train_conf.get('max_step', 10000)
    max_epoch = train_conf.get('max_epoch', 150)
    step_save_interval = train_conf.get('step_save_interval', 1000)
    epoch_save_interval = train_conf.get('epoch_save_interval', 1)
    log_interval = train_conf.get('log_interval', 100)
    use_amp = train_conf.get('use_amp', False)
    fp16_grad_sync = train_conf.get('fp16_grad_sync', False)

    # Set the training parameters
    save_model_dir = args.save_model_dir
    train_conf['rank'] = rank
    train_conf['world_size'] = world_size
    train_conf['max_keep_checkpoint'] = args.max_keep_checkpoint
    train_conf['start_epoch'] = start_epoch
    train_conf['start_batch'] = start_batch
    train_conf['is_distributed'] = distributed
    train_conf['use_amp'] = use_amp
    train_conf['save_model_dir'] = save_model_dir
    train_conf['cv_partition'] = cv_partition
    train_conf['timeout'] = args.timeout

    # Load the dataset and dataloader
    train_dataset, train_one_card_utts = Dataset(args.data_type,
                                                 args.train_data,
                                                 train_dataset_conf,
                                                 args.label_json,
                                                 args.timestamps,
                                                 partition=True,
                                                 whisper_processor=whisper_processor,
                                                 one_tar_nums=args.one_tar_nums)

    cv_dataset, cv_one_card_utts = Dataset(args.cv_data_type,
                                           args.cv_data,
                                           cv_dataset_conf,
                                           args.label_json,
                                           args.timestamps,
                                           partition=cv_partition,
                                           whisper_processor=whisper_processor,
                                           one_tar_nums=args.one_tar_nums)

    train_conf['train_one_card_utts'] = train_one_card_utts
    train_conf['cv_one_card_utts'] = cv_one_card_utts

    train_data_loader = DataLoader(train_dataset,
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

    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                num_workers=cv_dataloader_conf.get(
                                    'num_workers', 2),
                                pin_memory=cv_dataloader_conf.get(
                                    'pin_memory', True),
                                prefetch_factor=cv_dataloader_conf.get(
                                    'prefetch_factor', 4),
                                drop_last=cv_dataloader_conf.get(
                                    'drop_last', False),
                                )

    # Log the records
    writer = None
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f"{whisper_processor}")
        logger.info(f"{model}")
        logger.info(f"load prtrain model from: {args.pretrain_model_dir}")
        logger.debug(f"the number of model params: {num_params}")
        logger.info(f"save finetune model to: {save_model_dir}")

        os.makedirs(save_model_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(save_model_dir, 'tensorboard', time_string))

        saved_config_path = os.path.join(save_model_dir, 'train_init.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Register the model
    if distributed:
        # cuda model is required for nn.parallel.DistributedDataParallel
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False)

        if fp16_grad_sync:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default as comm_hooks,
            )
            model.register_comm_hook(
                state=None, hook=comm_hooks.fp16_compress_hook
            )
    else:
        model = model.to(device)

    # Register the optimizer
    if configs['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    elif configs['optim'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **configs['optim_conf'])
    else:
        raise ValueError("unknown optimizer: " + configs['optim'])
    if configs['scheduler'] == 'warmuplr':
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    elif configs['scheduler'] == 'NoamHoldAnnealing':
        scheduler = NoamHoldAnnealing(optimizer, **configs['scheduler_conf'])
    else:
        raise ValueError("unknown scheduler: " + configs['scheduler'])

    # used for pytorch amp mixed precision training
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Load the optimizer state from the last interruption
    if args.resume_train:
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
        if rank == 0:
            logger.warning(
                f"Optimizer state restored from the last interruption, current learning rate: {optimizer.param_groups[0]['lr']}")

    # Save the initial model
    if start_epoch == 0 and step == 0 and rank == 0:
        save_checkpoint_dir = os.path.join(
            save_model_dir, f"checkpoint_epoch_init")
        save_checkpoint(model, whisper_processor,
                        save_checkpoint_dir, optimizer, scheduler, infos={'max_keep_checkpoint': 1})

    # Initialize the trainer
    executor = Executor(step=step,
                        max_step=max_step,
                        max_epoch=max_epoch,
                        step_save_interval=step_save_interval,
                        epoch_save_interval=epoch_save_interval,
                        log_interval=log_interval,
                        metric_type=metric_type,
                        best_metric=best_metric,
                        use_smooth_loss=args.use_smooth_loss)
    if rank == 0:
        logger.debug(
            f"max_step: {max_step} max_epoch: {max_epoch} step_save_interval: {step_save_interval} "
            f"epoch_save_interval: {epoch_save_interval} log_interval: {log_interval} metric_type: {metric_type} "
            f"step: {step} start_epoch: {start_epoch} start_batch: {start_batch}")

    for epoch in range(start_epoch, max_epoch):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        if rank == 0:
            logger.debug(
                f"Training begin for epoch {executor.epoch}, current step: {executor.step} "
                f"total_train_utts: {train_one_card_utts*world_size} "
                f"train_one_card_utts: {train_one_card_utts} "
                f"total_cv_utts: {cv_one_card_utts*world_size if cv_partition else cv_one_card_utts} "
                f"cv_one_card_utts: {cv_one_card_utts}")
        executor.train(model, optimizer, scheduler, train_data_loader, cv_data_loader, device,
                    writer, train_conf, scaler, whisper_processor)
        if executor.epoch_save_interval and epoch + 1 % executor.epoch_save_interval == 0:
            executor.cv(model, cv_data_loader, device,
                        train_conf, whisper_processor, optimizer, scheduler)

        if executor.should_stop:
            break

    if rank == 0:
        writer.close()
        logger.info(
            f"Current epoch: {executor.epoch} step: {executor.step} Training completed!")


if __name__ == '__main__':
    args = get_args()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_dir = os.path.join(args.save_model_dir, "train_log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{time_string}.txt")
    logger = setup_logger('train_logger', log_path, level='DEBUG')
    main(args)
