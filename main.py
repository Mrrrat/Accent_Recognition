import wandb
import torch

from utils import seed_torch, count_parameters
from model import make_config, ClassificationNet, BatchOverfitModel
from dataloader import get_data_loaders
from train_eval import train
from augmentations import ComposeAugs, Volume, Fade, PitchShift, Noise


LABELS_PATH = 'meta.csv'
DATA_PATH = 'audios'


def main(config):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    seed_torch(80085)

    transform = ComposeAugs([Volume(p=0.25), Fade(p=0.25), PitchShift(p=0.25), Noise(p=0.2)], stretch_p=0.25)
    
    train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH, LABELS_PATH, batch_size=config['batch_size'])

    model = ClassificationNet(config).to(config['device'])
    #model = BatchOverfitModel(n_feats=config['n_feats'], n_class=config['n_class'], num_layers=config['num_layers'], ).to(config['device'])
    
    print(f'total params: {count_parameters(model)}')
    
    if config['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    wandb.init(project='Thesis', name=config['run_name'], config=config)
    train(config, model, optimizer, train_loader, val_loader)


if __name__ == '__main__':
    config = {
    'num_classes': 10, 
    'n_epochs': 20,
    'run_name': 'Supervised baseline Mel',
    'mode': 'mel',
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 64,
    'opt': 'AdamW',
    'num_workers': 8
    } 
    main(config)