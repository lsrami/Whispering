# Copyright (c) 2024 Timekettle Inc. (authors: Sirui Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from whispering.utils.checkpoint import save_checkpoint, save_checkpoint_best
from whispering.metrics import load_metric


class Executor:

    def __init__(self, step=0, max_step=10000, max_epoch=150, step_save_interval=100, epoch_save_interval=1, log_interval=10, metric_type = 'wer', best_metric=float('inf')):
        self.epoch = 0
        self.lr = 0.005
        self.should_stop = False
        self.cv_loss = 100
        self.train_loss = 100
        self.best_checkpoint_name = 'checkpoint_epoch_init'
        self.logger = logging.getLogger('train_logger')

        self.step = step
        self.max_step=max_step
        self.max_epoch=max_epoch
        self.step_save_interval=step_save_interval
        self.epoch_save_interval=epoch_save_interval
        self.log_interval=log_interval
        self.metric_type = metric_type
        self.best_metric = best_metric
        self.metric = load_metric(self.metric_type)


    def train(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, device,
                writer, train_conf, scaler, whisper_processor):
        ''' Train one epoch
        '''
        model.train()
        rank = train_conf.get('rank', 0)
        clip = train_conf.get('grad_clip', 50.0)
        accum_grad = train_conf.get('accum_grad', 1)
        start_epoch = train_conf.get('start_epoch', 0)
        start_batch = train_conf.get('start_batch', 0)

        is_distributed = train_conf.get('is_distributed', True)
        use_amp = train_conf.get('use_amp', False)
        if use_amp:
            self.logger.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(train_data_loader):
                if self.should_stop or self.step > self.max_step:
                    self.should_stop = True
                    break

                if self.epoch == start_epoch and batch_idx < start_batch:
                    continue
                keys, feats, labels = batch
                feats = feats.to(device)
                labels = labels.to(device)
                num_utts = labels.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        seq2seq_lm_output = model(input_features=feats, labels=labels)
                        loss = seq2seq_lm_output.loss
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                self.lr = optimizer.param_groups[0]['lr']
                self.train_loss = loss.item()
                num_seen_utts += num_utts

                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('loss/train', self.train_loss, self.step)
                        writer.add_scalar('batch/batch_size', num_utts, self.step)
                        writer.add_scalar('lr/lr', self.lr, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1

                # Record loss once every N log_intervals
                if rank == 0 and self.step % self.log_interval == 0:
                    log_str = f"Train-stage epoch: {self.epoch} step: {self.step} batch_idx: {batch_idx} train_loss: {self.train_loss} lr: {self.lr} num_seen_utts: {num_seen_utts}"
                    self.logger.info(log_str)

                # Validate and save the model once every N save_step_intervals
                if self.step % self.step_save_interval == 0:
                    self.cv(model, cv_data_loader, device, train_conf, whisper_processor, optimizer, scheduler)
                    if rank == 0:
                        writer.add_scalar(f'best_metric/{self.metric_type}', self.best_metric, self.step)
                        writer.add_scalar('loss/cv', self.cv_loss, self.step)


    def cv(self, model, data_loader, device, train_conf, whisper_processor, optimizer, scheduler):
        ''' Cross validation on
        '''
        model.eval()
        rank = train_conf.get('rank', 0)
        world_size = train_conf.get('world_size', 1)
        max_keep_checkpoint = train_conf.get('max_keep_checkpoint', 5)
        save_model_dir = train_conf.get('save_model_dir', 'save_model_dir')
        distributed = train_conf['is_distributed']
        forced_decoder_ids = train_conf['forced_decoder_ids']

        num_seen_utts = 0
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                keys, feats, labels = batch
                feats = feats.to(device)
                labels = labels.to(device)
                num_utts = labels.size(0)
                if num_utts == 0:
                    continue
                
                seq2seq_lm_output = model(input_features=feats, labels=labels)
                loss = seq2seq_lm_output.loss

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if rank == 0 and batch_idx % (self.log_interval*10) == 0:
                    log_str = f"CV-stage: epoch: {self.epoch} step: {self.step} batch_idx: {batch_idx} cv_loss: {total_loss / num_seen_utts} num_seen_utts: {num_seen_utts}"
                    self.logger.info(log_str)

                if self.metric_type != 'loss':
                    # Note: Convert -100 pads to tokenizer.pad_token_id before decoding.
                    labels = torch.where(labels != -100, labels, whisper_processor.tokenizer.pad_token_id)

                    # Note: In multi-language and multi-task hybrid training
                    # it is only verified on the specified task due to the limitation of forced_decoder_ids
                    model_instance = model.module if distributed else model
                    generated_preds = model_instance.generate(inputs=feats,
                                                            forced_decoder_ids=forced_decoder_ids,
                                                            max_new_tokens=448,
                                                            return_timestamps=False)


                    decoded_labels = whisper_processor.batch_decode(labels, skip_special_tokens=True)
                    decoded_preds = whisper_processor.batch_decode(generated_preds, skip_special_tokens=True)

                    self.metric.add_batch(references=decoded_labels, predictions=decoded_preds)


            num_seen_utts = 1 if num_seen_utts == 0 else num_seen_utts
            self.cv_loss = total_loss / num_seen_utts

            maximize_flag = False
            if self.metric_type != 'loss':
                current_metric = self.metric.compute()
                if self.metric_type == 'bleu':
                    current_metric = current_metric['bleu']
                    maximize_flag = True

            else:
                current_metric = self.cv_loss

            result_tensor = torch.tensor([self.cv_loss, current_metric], dtype=torch.float).to(device)
            if distributed:
                dist.reduce(result_tensor, dst=0, op=dist.ReduceOp.SUM)

            if rank == 0:
                result_tensor = result_tensor / world_size
                self.cv_loss, current_metric = result_tensor[0].item(), result_tensor[1].item()

                save_checkpoint_dir = os.path.join(save_model_dir, f"checkpoint_epoch{self.epoch}_step{self.step}")
                save_checkpoint(
                    model, whisper_processor, save_checkpoint_dir, optimizer, scheduler, {
                        'lr': self.lr,
                        'step': self.step,
                        'epoch': self.epoch,
                        'batch_idx': self.step,
                        'cv_loss': self.cv_loss,
                        'train_loss': self.train_loss,
                        'metric_type': self.metric_type,
                        'best_metric': current_metric,
                        'max_keep_checkpoint': max_keep_checkpoint
                    })

                best_checkpoint_dir = os.path.join(save_model_dir, f"checkpoint_best")
                is_best = save_checkpoint_best(current_metric, self.best_metric, save_checkpoint_dir, best_checkpoint_dir, maximize=maximize_flag)
                self.best_metric = current_metric if is_best else self.best_metric
                self.best_checkpoint_name = os.path.basename(save_checkpoint_dir) if is_best else self.best_checkpoint_name
                log_str = f"CV-done: epoch: {self.epoch} step: {self.step} cv_loss: {self.cv_loss} best_checkpoint: {self.best_checkpoint_name} metric_type: {self.metric_type} current_metric: {current_metric} best_metric: {self.best_metric} current_is_best: {is_best}"

                self.logger.debug(log_str)

