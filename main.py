import os
import wandb
import torch
from torch import nn
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking

from utils import seed_torch, count_parameters
from dataloader import get_data_loaders
from train_eval import train, test_epoch

from models import StupidModel, ECAPA_TDNN


LABELS_PATH = 'meta.csv'
DATA_PATH = 'audios'


def main(config):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    seed_torch(80085)

    #transform = None
    transform = nn.Sequential(
            # TimeStretch(n_freq=80, fixed_rate=0.8),
            FrequencyMasking(freq_mask_param=70),
            TimeMasking(time_mask_param=70)
    )
    
    train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH, LABELS_PATH, transform=transform, batch_size=config['batch_size'], num_workers=config['num_workers'], mode=config['mode'])

#     model = StupidModel(config['num_classes']).to(config['device'])
    model = ECAPA_TDNN(512, config['num_classes']).to(config['device'])
    #model = torch.compile(model)
    
    params = count_parameters(model)
    print(f'Total params: {params}')
    config['#params'] = params
    
    if config['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    wandb.init(project='Thesis', name=config['run_name'], config=config)
    train(config, model, optimizer, train_loader, val_loader)
    
    test_epoch(model, test_loader, config['device'], config['num_classes'])
    wandb.finish()


if __name__ == '__main__':
    config = {
    'num_classes': 9, 
    'n_epochs': 150,
    'run_name': 'ecapa-tdnn',
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 16,
    'opt': 'AdamW',
    'num_workers': 4,
    'smoothing': 0.1,
    'mode': 'wav'
    } 
    main(config)
