# train_dir = './input/data/train/'
# split_ratio = 0.2
# batch_size = 16

import os
import random
import argparse

import torch
import torch.nn as nn

import numpy as np

from adamp import AdamP
from sklearn.metrics import accuracy_score, f1_score

from model import MyModel
from inference import test
from dataset import get_dataloader
from inference import validation
from loss import F1Loss, FocalLoss, LabelSmoothingLoss


def str2bool(v):
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def training(model, optimizer, criterion, trainloader, validloader, Epochs, model_name):
    total_batch_ = len(trainloader)

    for i in range(1, Epochs + 1):
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
                print(f"Epoch {i:#04d} #{j + 1:#03d} "
                      f"loss: {batch_perform[0]/50:#.5f} || "
                      f"acc: {batch_perform[1]/50:#.2f} || "
                      f"f1 score: {batch_perform[2]/50:#.2f}")
                batch_perform = np.zeros(3)

            print(f"Epoch {i:#04d} "
                  f"loss: {epoch_perform[0]/total_batch_:#.5f} || "
                  f"acc: {epoch_perform[1]/total_batch_:#.2f} || "
                  f"f1 score: {epoch_perform[2]/total_batch_:#.2f}")

            validation(model, criterion, validloader)

            ###### Model save
            if i % 5 == 0:
                save_path = f'./models/{model_name}/chkpt-{i}.pt'
                torch.save(model.state_dict(), save_path)
                print("---------------- Saved checkpoint to: %s ----------------" % save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16,
                                help='input batch size for training (default: 16)')
    parser.add_argument('--model', type=str, default='resnet', help='model type (resnet; default or efficient)')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer type (sgd, adam; default, adamp)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                                help='criterion type (cross_entropy, f1, focal, label_smoothing)')

    parser.add_argument('--isTrain', type=str2bool, default=True, help='choose Train(true; default) or Test(false)')
    parser.add_argument('--train_dir', type=str, default='./input/data/train/images')
    parser.add_argument('--test_dir', type=str, default='./input/data/eval')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--chpkt_idx', type=int, default=10, help='checkpoint of models for submit')

    args = parser.parse_args()
    print(args)

    model = MyModel(model_type=args.model, n_classes=18)

    if args.isTrain:
        os.makedirs(f'./models/{args.model_name}', exist_ok=True)

        trainloader, validloader = get_dataloader(args.train_dir, args.val_ratio, args.batch_size)

        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optim == 'adamp':
            optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
        else:
            raise NameError('Not a optimizer available.')

        if args.loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif args.loss == 'f1':
            criterion = F1Loss()
        elif args.loss == 'focal':
            criterion = FocalLoss()
        elif args.loss == 'label_smoothing':
            criterion = LabelSmoothingLoss()
        else:
            raise NameError('Not a loss function available.')

        seed_everything(args.seed)
        training(model, optimizer, criterion, trainloader, validloader, args.epochs, args.model_name)

    else:

        test(model, args.model_name, str(args.chpkt_idx))
