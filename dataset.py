import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import Compose, ToTensor

def collate_fn(batch):
    # colours = [x[2] for x in batch]
    # colours = torch.nn.utils.rnn.pad_sequence(colours, batch_first=True, padding_value=-1)
    return (
        torch.stack([x[0] for x in batch]),
        torch.stack([x[1] for x in batch]),
        # colours
            )

class PixelImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, device='cpu'):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        self.filenames = glob.glob(img_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        x = np.load(self.filenames[idx])
        # image, label = x[0, ..., 0:1], x[1, ..., 0:1]
        image, label = x[0, 48:-48, 48:-48, 0:1], x[1, 48:-48, 48:-48, 0:1]
        # image, label = torch.tensor(image, device=self.device), torch.tensor(label, device=self.device)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            # colours = label.unique()[:, None, None]
            #filter colours
        # return [image, label, colours]
        return [image, label]

def get_dataloaders(dirpath, batch_size, val_ratio=0.1, num_workers=0):
    # transform = target_transform = None
    transform = Compose(
        [
            ToTensor(),
         ]
    )
    target_transform = Compose(
        [
            ToTensor(),
         ]
    )
    dataset = PixelImageDataset(dirpath, transform, target_transform)
    dataset_length = len(dataset)
    train_dataset = Subset(
        dataset,
        range(int(dataset_length*(1-val_ratio)))
    )
    val_dataset = Subset(
        dataset,
        range(int(dataset_length*(1-val_ratio)), dataset_length, 1)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size*2,
                                shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_dataloader, val_dataloader

