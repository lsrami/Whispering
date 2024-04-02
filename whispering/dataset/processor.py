# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

import io
import logging
import json
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
import torchaudio

torchaudio.set_audio_backend("sox_io")
torchaudio.utils.sox_utils.set_buffer_size(16500)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'txt':
                        example['txt'] = file_obj.read().decode('utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        # fix: AttributeError: '_Stream' object has no attribute 'seekable'
                        waveform, sample_rate = torchaudio.load(io.BytesIO(file_obj.read()), format=postfix)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        try:
            if 'start' in obj:
                assert 'end' in obj
                sample_rate = torchaudio.backend.sox_io_backend.info(
                    wav_file).sample_rate
                start_frame = int(obj['start'] * sample_rate)
                end_frame = int(obj['end'] * sample_rate)
                waveform, _ = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_file,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(key=key,
                           txt=txt,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def filter(data,
           max_length=30,
           min_length=1,
           token_max_length=448,
           token_min_length=1,
           min_output_input_ratio=0.003,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, feat, label, duration}]
            max_length: drop utterance which is greater than max_length(s)
            min_length: drop utterance which is less than min_length(s)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(s)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(s)

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'duration' in sample
        assert 'label' in sample

        duration = sample['duration']
        label_length = sample['label'].size(-1)

        if duration <= min_length:
            continue
        if duration > max_length:
            continue

        if label_length < token_min_length:
            continue
        if label_length > token_max_length:
            continue

        # sample['wav'] is torch.Tensor, we have 100 frames every second
        output_input_ratio = label_length / (duration * 100)
        if output_input_ratio < min_output_input_ratio:
            continue
        if output_input_ratio > max_output_input_ratio:
            continue

        del sample['duration']
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav

        yield sample


def _load_json_transcript(txt, timestamps):
    """Load JSON text data.

    Args:
        txt: JSON string containing language, task, and text information.
        timestamps: Boolean indicating whether timestamps are present in the data.

    Returns:
        Tuple containing language, task, and text.
    """
    try:
        txt_dict = json.loads(txt)
    except Exception as e:
        logging.warning(
            f"{e}\nInput: {txt}. Please confirm that the 'label_type' variable is correct.")
        return None, None, txt

    language = txt_dict.get('language', None)
    task = txt_dict.get('task', None)
    txt = txt_dict.get('sentence', '<|nospeech|>')
    if timestamps:
        sentences = txt_dict.get('sentences', [])
        if len(sentences):
            txt = sentences

    return language, task, txt


def _load_timestamps_transcript(whisper_processor, transcript, timestamp_begin, endoftext):
    """Load timestamps transcript.

    Args:
        whisper_processor: Whisper processor object.
        transcript: List of dictionaries containing 'start', 'end', and 'text'.
        timestamp_begin: Token representing the beginning of a timestamp.
        endoftext: Token representing the end of text.

    Returns:
        List of tokens representing the timestamps transcript.
    """
    labels = whisper_processor.tokenizer.prefix_tokens[:3]

    for t in transcript:
        # note: round(t['start'] * 100) // 2 corresponds to the output frame rate of 20ms for the Whisper model.
        start = t['start'] if round(
            t['start'] * 100) % 2 == 0 else t['start'] + 0.01
        start = timestamp_begin + round(start * 100) // 2
        end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
        end = timestamp_begin + round(end * 100) // 2
        label = whisper_processor(text=t['text']).input_ids[4:-1]

        labels.append(start)
        labels.extend(label)
        labels.append(end)

    labels.append(endoftext)
    return labels


def data_processor(data, whisper_processor, label_json=False, timestamps=False):
    """ Apply data processor.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]
            whisper_processor: Whisper processor object.
            label_json: Whether to use json format labels.
            timestamps: Boolean indicating whether timestamps are present in the data.

        Returns:
            Iterable[{key, feat, label, duration}]
    """
    vocab = whisper_processor.tokenizer.get_vocab()
    startoftranscript = vocab['<|startoftranscript|>']
    endoftext = vocab['<|endoftext|>']
    # Compatibility with old and new models for 'nospeech' and timestamp tokens
    nospeech = vocab.get('<|nospeech|>', vocab.get('<|nocaptions|>'))
    timestamp_begin = vocab.get('<|0.00|>', vocab.get('<|notimestamps|>') + 1)

    for sample in data:
        assert 'key' in sample
        assert 'wav' in sample
        assert 'txt' in sample
        assert 'sample_rate' in sample

        txt = sample['txt']
        waveform = sample['wav'].squeeze()
        sample_rate = sample['sample_rate']
        duration = waveform.size(-1) / sample_rate

        if label_json:
            language, task, txt = _load_json_transcript(txt, timestamps)
        else:
            language, task = None, None


        # Set language and task for each individual entry
        if language:
            if task:
                whisper_processor.tokenizer.set_prefix_tokens(
                    task=task, language=language)
            else:
                whisper_processor.tokenizer.set_prefix_tokens(
                    language=language)
        elif task:
            whisper_processor.tokenizer.set_prefix_tokens(task=task)

        # 按照任务类型做tokenizer
        if isinstance(txt, list):  # with timestamps
            inputs = whisper_processor(
                audio=waveform, sampling_rate=sample_rate, return_attention_mask=False, return_tensors="pt")
            labels = _load_timestamps_transcript(
                whisper_processor, txt, timestamp_begin, endoftext)
            inputs['labels'] = torch.tensor(labels, dtype=torch.long)
        else:
            if txt == '<|nospeech|>' or len(txt) == 0:
                inputs = whisper_processor(
                    audio=waveform, sampling_rate=sample_rate, return_attention_mask=False, return_tensors="pt")
                inputs['labels'] = torch.tensor(
                    [startoftranscript, nospeech, endoftext], dtype=torch.long)

            else:
                inputs = whisper_processor(audio=waveform, sampling_rate=sample_rate,
                                           text=txt, return_attention_mask=False, return_tensors="pt")

        feat = inputs['input_features'].squeeze(0).permute(1, 0)

        label = inputs['labels'].squeeze(0)

        yield dict(key=sample['key'], feat=feat, label=label, duration=duration)


def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def spec_sub(data, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, whisper_processor):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels)]
    """
    decoder_start_token_id = whisper_processor.tokenizer.get_vocab()[
        '<|startoftranscript|>']
    for sample in data:
        assert 'key' in sample[0]
        assert 'feat' in sample[0]
        assert 'label' in sample[0]

        keys = [s['key'] for s in sample]

        feats = torch.stack([s['feat'].permute(1, 0) for s in sample])

        label_features = [{"input_ids": s['label']} for s in sample]
        labels_batch = whisper_processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # Note: The shift_tokens_right method, which automatically adds decoder_start_token_id,
        # has already been implemented in the model's forward method. The decoder_start_token_id needs to be removed first.
        # In the configuration file provided by the official,
        # the bos_token_id is incorrectly equal to eos_token_id, so here it is changed to use decoder_start_token_id.
        if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        yield (keys, feats, labels)
