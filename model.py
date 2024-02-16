import lightning.pytorch as pl
import torch
import torch.nn as nn
import os
from modules import FirstLevelNet, SecondLevelNet

class CoGCNet(pl.LightningModule):
    def __init__(self, first_level_model, second_level_model, **params):
        super().__init__()
        
        self.first_level_model = first_level_model
        self.second_level_model = second_level_model
        self.params = params
        self.criterion = nn.CrossEntropyLoss()
        
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        spectrograms, labels = batch
        output = self.first_level_model(spectrograms)
        preds = self.second_level_model(output)
        train_loss = self.criterion(preds, labels)
        
        self.log("train_loss", train_loss.data)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        spectrograms, labels = batch
        output = self.first_level_model(spectrograms)
        preds = self.second_level_model(output)
        val_loss = self.criterion(preds, labels)
        
        self.log("val_loss", val_loss.data)
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.first_level_model.parameters(), 'lr': self.params['lr']},
            {'params': self.second_level_model.parameters(), 'lr': self.params['lr']}
            ])
        
        return self.optimizer
        
        



        
