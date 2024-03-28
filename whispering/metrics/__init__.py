from .bleu.bleu import Bleu
from .cer.cer import CER
from .wer.wer import WER


def load_metric(metric_type):
    if metric_type == 'cer':
        return CER()
    elif metric_type == 'wer':
        return WER()
    elif metric_type == 'bleu':
        return Bleu()
    elif metric_type == 'loss':
        return metric_type
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")