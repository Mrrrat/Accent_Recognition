import os
import torch
from typing import List
from torch.nn.functional import pad
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class AccentDataset(Dataset):
    def __init__(self, meta, audio_dir, idx=None, transform=None):
        super().__init__()
        if idx is None:
            idx = np.arange(len(meta.index), dtype=int)
        self.audios = pd.read_csv(meta).loc[idx, 'path'].values
        self.targets = pd.read_csv(meta).loc[idx, 'target'].values
        self.audio_dir = audio_dir
        self.transform = None
        if transform is not None:
            self.transform = transform

    def get_item(self, idx):
        path = os.path.join(self.audio_dir, self.audios[idx])
        waveform, sample_rate = torchaudio.load(path)
        if self.transform is not None:
            waveform = self.transform(waveform)
        target = self.targets[idx]
        return waveform.squeeze(0), target

    def __len__(self):
        return len(self.audios)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here
    for key in ['text_encoded', 'spectrogram', 'audio']:
        max_len = max(list(map(lambda x: x[key].shape[-1], dataset_items)))
        result_batch[key] = torch.cat(list(
            map(lambda x: pad(x[key], (0, max_len - x[key].shape[-1])), dataset_items)
        ), dim=0)

    for key in ['text', 'duration', 'audio_path']:
        result_batch[key] = [x[key] for x in dataset_items]

    result_batch['spectrogram'] = result_batch['spectrogram']
    result_batch['text_encoded_length'] = torch.tensor(list(map(lambda x: x['text_encoded'].shape[-1], dataset_items)),
                                                       dtype=torch.int32)
    result_batch['spectrogram_length'] = torch.tensor(list(map(lambda x: x['spectrogram'].shape[-1], dataset_items)),
                                                      dtype=torch.int32)
    return result_batch


def get_data_loaders(data_dir, meta_path, transform=None, bs=512):
    meta = pd.read_csv(meta_path)
    train_idx, val_idx, _, _ = train_test_split(np.arange(meta.shape[0], dtype=int), meta['target'], test_size=0.2,
                                                stratify=meta['target'])
    train_dataset = AccentDataset(data_dir, meta_path, train_idx, transform=transform)
    val_dataset = AccentDataset(data_dir, meta_path, val_idx, transform=transform)

    # TODO
    weights = 1.0 / meta.loc[train_idx, 'target_frequency'].values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))

    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=0, pin_memory=True,
                              drop_last=True, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True)
    return train_loader, val_loader
