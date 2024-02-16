import os
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as transforms
import lightning.pytorch as pl

class AudioDataset(Dataset):
    def __init__(self, data_config, mode='train'):
        super().__init__()
        assert mode in ['train', 'val']
        if mode == 'train':
            self.data_dir = data_config['train_path']
        elif mode =='val':
            self.data_dir = data_config['val_path']
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.audio_files = self._load_audio_files()

    def _load_audio_files(self):
        audio_files = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for file in os.listdir(class_dir):
                audio_files.append((os.path.join(class_dir, file), class_name))
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file, class_name = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        # stereo to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0)
        return waveform, self.class_to_idx[class_name]

class MelSpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config

    def setup(self, stage=None):
        self.train_dataset = AudioDataset(self.data_config, mode='train')
        self.val_dataset = AudioDataset(self.data_config, mode='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.data_config['batchsize'], 
                          num_workers=10, collate_fn=self.collate_fn, pin_memory=True, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.data_config['batchsize'], 
                          num_workers=10, collate_fn=self.collate_fn, pin_memory=True, shuffle=False)
        

    def collate_fn(self, batch):
        waveforms, labels = zip(*batch)
        mel_transform = transforms.MelSpectrogram(
            sample_rate = self.data_config['sr'],
            n_fft = self.data_config['n_fft'],
            n_mels = self.data_config['n_mels']
        )
        
        mel_spectrograms = []
        for waveform in waveforms:
            mel_spectrogram = mel_transform(waveform)
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
            mel_spectrograms.append(mel_spectrogram)
        return torch.stack(mel_spectrograms), torch.tensor(labels)

if __name__ == "__main__":
    with open('config.yaml', 'r') as confread:
        try:
            config = yaml.safe_load(confread)['data']
        except yaml.YAMLError as e:
            print(e)
            
    mel_datamodule = MelSpectrogramDataModule(config)
    mel_datamodule.setup()
    train_loader = mel_datamodule.train_dataloader()

    for batch in train_loader:
        # batch contains mel spectrograms and labels
        mel_spectrograms, labels = batch
        print(mel_spectrograms.shape)  # Shape will be [batch_size, channels, frequency_bins, time_frames]
        print(labels)  # Tensor containing label indices
