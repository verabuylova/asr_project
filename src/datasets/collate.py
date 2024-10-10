import logging
from typing import List

import torch

logger = logging.getLogger(__name__)

def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    audio = [item['audio'] for item in dataset_items]

    spectrograms = [item['spectrogram'].squeeze(0) for item in dataset_items]
    spectrograms = [spec.T for spec in spectrograms]
    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms)
    spectrograms = spectrograms.permute(1, 2, 0) 
    spectrograms = torch.log(spectrograms + 1e-10)
    
    texts = [item['text'] for item in dataset_items]

    texts_encoded = [item['text_encoded'] for item in dataset_items]
    texts_encoded = [tex.T for tex in texts_encoded]
    texts_encoded = torch.nn.utils.rnn.pad_sequence(texts_encoded)
    texts_encoded = texts_encoded.permute(1, 2, 0).squeeze(1)

    texts_encoded_lens = torch.tensor([text_encoded.shape[-1] for text_encoded in texts_encoded])

    spectrogram_lens = torch.tensor([spec.shape[1] for spec in spectrograms])

    audio_paths = [item['audio_path'] for item in dataset_items] 
    

    return {
        'spectrogram': spectrograms,
        'spectrogram_length': spectrogram_lens,
        'text_encoded': texts_encoded,
        'text_encoded_length': texts_encoded_lens,
        'text': texts,
        'audio_path': audio_paths,
        'audio': audio
    }