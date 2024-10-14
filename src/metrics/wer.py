from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

class BeamSearchWERMetric(BaseMetric):
    def __init__(
        self, text_encoder, type: str = "BS", beam_size=10, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.type = type
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, probs: Tensor, logits: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        probs_ = probs.detach().cpu().numpy()
        lengths = log_probs_length.detach().numpy()
        for prob, length, target_text in zip(probs_, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_beam_search(False, log_probs, prob[:length], logits, 10)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

class BeamSearchLMWERMetric(BaseMetric):
    def __init__(
        self, text_encoder, beam_size=100, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.type = type
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, probs: Tensor, logits: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        logits_ = logits.detach().cpu().numpy()
        lengths = log_probs_length.detach().numpy()
        for logit, length, target_text in zip(logits_, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_beam_search(True, log_probs, probs, logit[:length], 100)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
    