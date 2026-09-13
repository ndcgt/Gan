import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class AnimeDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.images = []

        if self.train:
            subset = 'train'
        else:
            subset = 'test'

        folder = os.path.join(self.root, subset)
        for filename in os.listdir(folder):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                self.images.append(os.path.join(folder, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img