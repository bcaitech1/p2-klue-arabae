import os
import glob
import random
import pandas as pd
from PIL import Image

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop, RandomHorizontalFlip

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


def test(model, test_dir, model_name, chpkt_idx):
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    transform = transforms.Compose([
        Resize((512, 384), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])

    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )

    device = torch.device('cuda')
    load_path = f'./models/{model_name}/chkpt-{chpkt_idx}.pt'
    model.load_state_dict(torch.load(load_path))
    model = model.cuda()

    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in loader:
        with torch.no_grad():
            images = images.float().to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(f'./results/{model_name}_epoch{chpkt_idx}.csv', index=False)
    print('test inference is done!')
