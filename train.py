from modules import FirstLevelNet, SecondLevelNet
from model import CoGCNet
from dataset import MelSpectrogramDataModule
import yaml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

with open('config.yaml', 'r') as confread:
    try:
        config = yaml.safe_load(confread)
    except yaml.YAMLError as e:
        print(e)
        
first_level_model = FirstLevelNet(**config['model']['firstlevel']).cuda()
second_level_model = SecondLevelNet(**config['model']['secondlevel']).cuda()
train_config = config['train']
data_config = config['data']
model = CoGCNet(first_level_model, second_level_model, **train_config)

logger = TensorBoardLogger(**train_config['logging'])

seed_everything(train_config['manual_seed'], True)


data = MelSpectrogramDataModule(data_config)
data.setup()

trainer = Trainer(logger=logger,**train_config['trainer'])

trainer.fit(model, datamodule=data)

