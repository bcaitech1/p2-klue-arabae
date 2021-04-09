#!/usr/bin/env python
# coding: utf-8

import os
import glob
import random
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def seed_everything(seed):
    """
    동일한 조건으로 학습을 할 때, 동일한 결과를 얻기 위해 seed를 고정시킵니다.
    
    Args:
        seed: seed 정수값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(42)


### Loss
# -- Focal Loss
# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

# -- Label Smoothing Loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

    
# -- F1 Loss
# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
    

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


train_dir = './input/data/train/'
train_pd = pd.read_csv(os.path.join(train_dir, 'train.csv'))
train_paths, valid_paths = train_test_split(train_pd.path, test_size=0.2, shuffle=True)

# Train Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
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
    batch_size=16,
    shuffle=True
)

validloader = DataLoader(
    vailddataset,
    shuffle=False
)


### Model
n_classes = 18
model = torchvision.models.resnet50(pretrained=True)
'''
for param in model.parameters():
    param.requires_grad = False
'''

# inplace를 true로하면 input으로 들어온 것 자체를 수정
model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes),
        )

Epochs = 30
learning_rate = 1e-5

model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = AdamP(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

### Training
model_name = 'baseline_ver11'
model_dir = f'./models/{model_name}'
os.makedirs(model_dir, exist_ok=True)


total_batch_ = len(trainloader)
valid_batch_ = len(validloader)

for i in range(1, Epochs+1):
    model.train()
    epoch_perform, batch_perform = np.zeros(3), np.zeros(3)
    
    for j, (images, labels) in enumerate(trainloader):
        images, labels = images.float().cuda(), labels.cuda()
        
        optimizer.zero_grad()
        output = model(images)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        predict = output.argmax(dim=-1)
        predict = predict.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        acc = accuracy_score(labels, predict)
        f1 = f1_score(labels, predict, average='macro')
        
        batch_perform += np.array([loss.item(), acc, f1])
        epoch_perform += np.array([loss.item(), acc, f1])
        
        if (j + 1) % 50 == 0:
            print(f"Epoch {i:#04d} #{j+1:#03d} -- loss: {batch_perform[0]/50:#.5f}, acc: {batch_perform[1]/50:#.2f}, f1 score: {batch_perform[2]/50:#.2f}")
            batch_perform = np.zeros(3)
            
    print(f"Epoch {i:#04d} loss: {epoch_perform[0]/total_batch_:#.5f}, acc: {epoch_perform[1]/total_batch_:#.2f}, f1 score: {epoch_perform[2]/total_batch_:#.2f}")
    
    ###### Validation
    model.eval()
    valid_perform = np.zeros(3)
    with torch.no_grad():
        for valid_images, valid_labels in validloader:
            valid_images, valid_labels = valid_images.float().cuda(), valid_labels.cuda()
            
            valid_output = model(valid_images)
            valid_loss = criterion(valid_output, valid_labels)
            
            valid_predict = valid_output.argmax(dim=-1)
            valid_predict = valid_predict.detach().cpu().numpy()
            valid_labels = valid_labels.detach().cpu().numpy()

            valid_acc = accuracy_score(valid_labels, valid_predict)
            valid_f1 = f1_score(valid_labels, valid_predict, average='macro')
            
            valid_perform += np.array([valid_loss.item(), valid_acc, valid_f1])
            
        print(f">>>> Validation loss: {valid_perform[0]/valid_batch_:#.5f}, Acc: {valid_perform[1]/valid_batch_:#.2f}, f1 score: {valid_perform[2]/valid_batch_:#.2f}")
        print()
        
    ###### Model save
    if i % 5 == 0:
        save_path = f'./models/{model_name}/chkpt-{i}.pt'
        torch.save(model.state_dict(), save_path)
        print("---------------- Saved checkpoint to: %s ----------------" % save_path)
