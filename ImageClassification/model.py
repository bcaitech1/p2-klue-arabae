import timm

import torch
import torch.nn as nn

import torchvision


class MyModel():
    def __init__(self, model_type, n_classes):
        if model_type == 'resnet':
            self.model = torchvision.models.resnet50(pretrained=True)

            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, n_classes),
            )

        elif model_type == 'efficient':
            self.model = timm.create_model('tf_efficientnet_b4', pretrained=True)

            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, n_classes),
            )

    def forward(self, x):
        x = self.model(x)
        return x
