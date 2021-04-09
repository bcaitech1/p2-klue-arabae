import os
import glob
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

from sklearn.model_selection import train_test_split


### Dataset
class TrainDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.get_label(self.img_paths[index])

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_paths)

    def get_label(self, p):
        path, file = p.split('/')[-2], p.split('/')[-1]
        label_key = 0
        if 'incorrect' in file:
            label_key = 6
        elif 'normal' in file:
            label_key = 12

        if 'female' in path:
            label_key += 3

        age = int(path[-2:])
        if age >= 58:
            label_key += 2
        elif  age >= 30 and age <58:
            label_key += 1

        return label_key


def get_dataloader(train_dir, split_ratio, batch_size):
    train_pd = pd.read_csv(os.path.join(train_dir, 'train.csv'))
    train_paths, valid_paths = train_test_split(train_pd.path, test_size=split_ratio, shuffle=True)

    train_samples = [glob.glob(os.path.join(train_dir, 'images' + f'/{t}/*'), recursive=True) for t in train_paths]
    valid_samples = [glob.glob(os.path.join(train_dir, 'images' + f'/{v}/*'), recursive=True) for v in valid_paths]
    train_samples, valid_samples = sum(train_samples, []), sum(valid_samples, [])

    train_transform = transforms.Compose([
        Resize((512, 384), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])

    valid_transform = transforms.Compose([
        Resize((512, 384), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])

    traindataset = TrainDataset(train_samples, train_transform)
    vailddataset = TrainDataset(valid_samples, valid_transform)

    trainloader = DataLoader(
        traindataset,
        batch_size=batch_size,
        shuffle=True
    )

    validloader = DataLoader(
        vailddataset,
        shuffle=False
    )

    return trainloader, validloader
