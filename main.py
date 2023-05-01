import os
import wandb
import torch
from torch import nn
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking

from utils import seed_torch, count_parameters
from dataloader import get_data_loaders
from train_eval import train, test_epoch
import transformers

from models import StupidModel, ECAPA_TDNN, AccentTransformer, ClassificationNet


LABELS_PATH = 'meta.csv'
DATA_PATH = 'audios'


def main(config):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    seed_torch(80085)

    #transform = None
    transform = nn.Sequential(
            # TimeStretch(n_freq=80, fixed_rate=0.8),
            FrequencyMasking(freq_mask_param=80),
            TimeMasking(time_mask_param=80)
    )
    
    train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH, LABELS_PATH, transform=transform, batch_size=config['batch_size'], num_workers=config['num_workers'], mode=config['mode'], n_mels=config['n_mels'])
    
#     model = StupidModel(config['num_classes']).to(config['device'])
#     model = ECAPA_TDNN(512, config['num_classes']).to(config['device'])
#     model = AccentTransformer(num_classes=config['num_classes'], emb_size=128, hidden_size=512, n_layers=4, n_head=4, dropout=0.1).to(config['device'])
    model = ClassificationNet(num_classes=config['num_classes'], hidden_dim=1024, attn_dim=512, n_mels=config['n_mels']).to(config['device'])
    #model = torch.compile(model)
    
    params = count_parameters(model)
    print(f'Total params: {params}')
    config['#params'] = params
    
    if config['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    
    scheduler = None
#     scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=15, num_training_steps=config['n_epochs'] * len(train_loader), num_cycles=0.5)
    
    wandb.init(project='Thesis', name=config['run_name'], config=config)
    train(config, model, optimizer, train_loader, val_loader, scheduler=scheduler)
    
    test_epoch(model, test_loader, config['device'], config['num_classes'])
    wandb.finish()


if __name__ == '__main__':
    config = {
    'num_classes': 9, 
    'n_epochs': 150,
    'run_name': 'QuartzNet',
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 16,
    'n_mels': 80,
    'opt': 'AdamW',
    'num_workers': 4,
    'smoothing': 0.1,
    'mode': 'mel'
    } 
    main(config)
