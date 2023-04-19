import os
import pandas as pd
import numpy as np

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from utils import LogMelSpectrogram


class AccentDataset(Dataset):
    def __init__(self, audio_dir, meta, idx=None, transform=None, freq=16000):
        super().__init__()
        if idx is None:
            idx = np.arange(len(meta.index), dtype=int)
        self.audios = pd.read_csv(meta).loc[idx, 'path'].values
        self.targets = pd.read_csv(meta).loc[idx, 'target'].values
        self.audio_dir = audio_dir
        self.transform = transform
        self.freq = freq

    def __getitem__(self, idx):
        path = os.path.join(self.audio_dir, self.audios[idx] + '.wav')
        waveform, sample_rate = torchaudio.load(path)
        waveform = T.Resample(orig_freq=sample_rate, new_freq=self.freq)(waveform)
        
        spectrogram = LogMelSpectrogram(self.freq, n_mels=80)
        spectrogram = spectrogram(waveform.squeeze())
        
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        target = self.targets[idx]
        return spectrogram, target

    def __len__(self):
        return len(self.audios)


def collate_fn(batch):
    max_length = 0
    for sample, label in batch:
        max_length = max(max_length, sample.size(-1))
    samples, labels = torch.zeros((len(batch), 1, sample.size(0), max_length)).fill_(value=-11.52), []
    for i, (sample, label) in enumerate(batch):
        samples[i, :, :, :sample.size(-1)] = sample.unsqueeze(0)
        labels.append(label)
    labels = torch.LongTensor(labels)
    return samples, labels


def get_data_loaders(data_dir, meta_path, transform=None, batch_size=64, num_workers=8):
    meta = pd.read_csv(meta_path)
    train_idx, val_idx, _, _ = train_test_split(np.arange(meta.shape[0], dtype=int), meta['target'], test_size=0.2,
                                                stratify=meta['target'])
    val_idx, test_idx, _, _ = train_test_split(val_idx, meta['target'][val_idx], test_size=0.5,
                                                stratify=meta['target'][val_idx])
    
    train_dataset = AccentDataset(data_dir, meta_path, train_idx, transform=transform)
    val_dataset = AccentDataset(data_dir, meta_path, val_idx, transform=transform)
    test_dataset = AccentDataset(data_dir, meta_path, test_idx, transform=transform)

    weights = 1.0 / meta.loc[train_idx, 'count'].values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
                              drop_last=True, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
