import os
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class AccentDataset(Dataset):
    def __init__(self, audio_dir, meta, idx=None, transform=None):
        super().__init__()
        if idx is None:
            idx = np.arange(len(meta.index), dtype=int)
        self.audios = pd.read_csv(meta).loc[idx, 'path'].values
        self.targets = pd.read_csv(meta).loc[idx, 'target'].values
        self.audio_dir = audio_dir
        self.transform = None
        if transform is not None:
            self.transform = transform

    def __getitem__(self, idx):
        path = os.path.join(self.audio_dir, self.audios[idx])
        waveform, sample_rate = torchaudio.load(path)
        if self.transform is not None:
            waveform = self.transform(waveform)
        target = self.targets[idx]
        return waveform.squeeze(0), target

    def __len__(self):
        return len(self.audios)


def get_data_loaders(data_dir, meta_path, transform=None, bs=512):
    meta = pd.read_csv(meta_path)
    train_idx, val_idx, _, _ = train_test_split(np.arange(meta.shape[0], dtype=int), meta['target'], test_size=0.2,
                                                stratify=meta['target'])
    train_dataset = AccentDataset(data_dir, meta_path, train_idx, transform=transform)
    val_dataset = AccentDataset(data_dir, meta_path, val_idx, transform=transform)

    weights = 1.0 / meta.loc[train_idx, 'count'].values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))

    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=0, pin_memory=True,
                              drop_last=True, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True)
    return train_loader, val_loader
