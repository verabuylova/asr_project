from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            if met.name in ["CER_(BS)", "WER_(BS)", "CER_(BS_LM)", "WER_(BS_LM)"]:
                if self.current_epoch % 10 == 0: 
                    metrics.update(met.name, met(**batch))
            else:
                metrics.update(met.name, met(**batch))

        return batch


    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, text, log_probs, probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly


        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        preds_bs = [self.text_encoder.ctc_beam_search(False, log_probs, probs_element[:log_probs_length_element], 4) 
                    for (probs_element, log_probs_length_element) in zip(probs, log_probs_length)]
        
        preds_bs_lm = [self.text_encoder.ctc_beam_search(True, log_probs, probs_element[:log_probs_length_element], 50) 
                    for (probs_element, log_probs_length_element) in zip(probs, log_probs_length)]
        

        tuples = list(zip(argmax_texts, preds_bs, preds_bs_lm, text, argmax_texts_raw, audio_path))

        rows = {}
        for pred, pred_bs, pred_bs_lm, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            wer_bs = calc_wer(target, pred_bs) * 100
            cer_bs = calc_cer(target, pred_bs) * 100

            wer_bs_lm = calc_wer(target, pred_bs_lm) * 100
            cer_bs_lm = calc_cer(target, pred_bs_lm) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "predictions bs": pred_bs,
                "predictions bs+lm": pred_bs_lm,
                "wer": wer,
                "cer": cer,
                "wer_bs": wer_bs,
                "cer_bs": cer_bs,
                "wer_bs_lm": wer_bs_lm,
                "cer_bs_lm":cer_bs_lm
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
