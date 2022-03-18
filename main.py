import wandb
import torch

from utils import seed_torch, count_parameters
from model import make_config, ClassificationNet, BatchOverfitModel
from dataloader import get_data_loaders
from train_eval import train
from augmentations import ComposeAugs, Volume, Fade, PitchShift, Noise


LABELS_PATH = 'D:/Downloads/archive/meta5.csv'
DATA_PATH = 'D:/Downloads/archive/recordings/recordings/'


def main():
	SEED = 67
	config = make_config()
	seed_torch(SEED)

	transform = ComposeAugs([Volume(p=0.25), Fade(p=0.25), PitchShift(p=0.25), Noise(p=0.2)], stretch_p=0.25)

	train_loader, val_loader = get_data_loaders(DATA_PATH, LABELS_PATH, transform=transform, bs=64)

	wandb.init(project='Course Work', name='first try', config=config)

	#model = ClassificationNet(config).to(config['device'])
	model = BatchOverfitModel(n_feats=config['n_feats'], n_class=config['n_class'], num_layers=config['num_layers'], ).to(config['device'])
	print(f'total params: {count_parameters(model)}')
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
	train(config, model, optimizer, train_loader, val_loader)


if __name__ == '__main__':
	main()