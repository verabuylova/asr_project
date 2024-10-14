import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    text_encoded_lengths = torch.tensor([item["text_encoded"].shape[1] for item in dataset_items], dtype=torch.long)
    spectrogram_lengths = torch.tensor([item["spectrogram"].shape[2] for item in dataset_items], dtype=torch.long)

    spectrograms = pad_sequence(
        [item["spectrogram"].squeeze(0).transpose(0, -1) for item in dataset_items],
        batch_first=True,
        padding_value=0.0
    ).transpose(1, -1)

    text_encodeds = pad_sequence(
        [item["text_encoded"].squeeze(0).transpose(0, -1) for item in dataset_items],
        batch_first=True,
        padding_value=0
    ).transpose(1, -1)

    audios = pad_sequence(
        [item["audio"].squeeze(0) for item in dataset_items],
        batch_first=True,
        padding_value=0.0
    ).transpose(1, -1)

    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    result_batch = {
        "text_encoded_length": text_encoded_lengths,
        "spectrogram_length": spectrogram_lengths,
        "spectrogram": spectrograms,
        "text_encoded": text_encodeds,
        "text": texts,
        "audio": audios,
        "audio_path": audio_paths,
    }

    return result_batch
