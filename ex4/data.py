from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, dataframe, mode):
        super().__init__()

        self.data = dataframe
        self.mode = mode

        self.transform_train = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])

        self.transform_val = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get your data from the dataframe here
        image_p = Path(self.data.iloc[index,0])
        label = np.array(self.data.iloc[index, 1:]).astype('float')

        # Convert grayscale image to RGB
        image = imread(image_p)

        image = gray2rgb(image)

        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_val(image)

        return image, torch.tensor(label, dtype=torch.long)