import os
import random
import argparse
import numpy as np
import sys

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

from dataloader import load_data, get_trainLoader
from model import BERTClassifier
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel

sys.path.append('/opt/ml/klue-baseline')
from evaluation import test_main

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


def train(model, optimizer, criterion, trainloader, validloader, Epochs, model_name, wandb):
    total_batch_ = len(trainloader)
    valid_batch_ = len(validloader)

    model.cuda()

    for i in range(Epochs + 1):
        model.train()
        epoch_perform, batch_perform = np.zeros(2), np.zeros(2)

        for j, v in enumerate(trainloader):
            input_ids, attention_mask, token_type_ids, labels = v['input_ids'].cuda(), v['attention_mask'].cuda(), v[
                'token_type_ids'].cuda(), v['labels'].cuda()
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, token_type_ids)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            predict = output.argmax(dim=-1)
            predict = predict.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            acc = accuracy_score(labels, predict)

            batch_perform += np.array([loss.item(), acc])
            epoch_perform += np.array([loss.item(), acc])

            if (j + 1) % 50 == 0:
                print(
                    f"Epoch {i:#04d} #{j + 1:#03d} -- loss: {batch_perform[0] / 50:#.5f}, acc: {batch_perform[1] / 50:#.2f}")
                batch_perform = np.zeros(2)
        print(
            f"Epoch {i:#04d} loss: {epoch_perform[0] / total_batch_:#.5f}, acc: {epoch_perform[1] / total_batch_:#.2f}")
        wandb.log({
            "Train epoch Loss": epoch_perform[0] / total_batch_,
            "Train epoch Acc": epoch_perform[1] / total_batch_})
        ###### Validation

        model.eval()
        valid_perform = np.zeros(2)
        with torch.no_grad():
            for v in validloader:
                input_ids, attention_mask, token_type_ids, valid_labels = v['input_ids'].cuda(), v['attention_mask'].cuda(), \
                                                                    v['token_type_ids'].cuda(), v['labels'].cuda()
                valid_output = model(input_ids, attention_mask, token_type_ids)
                valid_loss = criterion(valid_output, valid_labels)

                valid_predict = valid_output.argmax(dim=-1)
                valid_predict = valid_predict.detach().cpu().numpy()
                valid_labels = valid_labels.detach().cpu().numpy()

                valid_acc = accuracy_score(valid_labels, valid_predict)

                valid_perform += np.array([valid_loss.item(), valid_acc])

            print(
                f">>>> Validation loss: {valid_perform[0] / valid_batch_:#.5f}, Acc: {valid_perform[1] / valid_batch_:#.2f}")
            print()
            wandb.log({
                "Valid Loss": valid_perform[0] / valid_batch_,
                "Valid Acc": valid_perform[1] / valid_batch_})

        ###### Model save
        if i % 5 == 0:
            save_path = f'./models/{model_name}/chkpt-{i}.pt'
            torch.save(model.state_dict(), save_path)
            print("---------------- Saved checkpoint to: %s ----------------" % save_path)



if __name__ == '__main__':
    import wandb
    wandb.init()

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--token', type=str, default='kobert', help='tokenizer type (kobert; default)')
    parser.add_argument('--model', type=str, default='kobert', help='model type (kobert; default)')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer type (sgd, adam; default, adamp)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 5e-5)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio for validaton (default: 0.1)')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        help='criterion type (cross_entropy)')

    parser.add_argument('--isTrain', type=str2bool, default=True, help='choose Train(true; default) or Test(false)')
    parser.add_argument('--train_dir', type=str, default='../input/data/train')
    parser.add_argument('--test_dir', type=str, default='../input/data/test')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--chpkt_idx', type=int, default=10, help='checkpoint of models for submit')

    args = parser.parse_args()
    print(args)
    wandb.run.name = f'{args.model_name}'
    wandb.config.update(args)

    if args.token == 'kobert':
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    else:
        raise NameError('Not a tokenizer available.')

    if args.model == 'kobert':
        pretrained_model = BertModel.from_pretrained("monologg/kobert")
        model = BERTClassifier(pretrained_model)
    else:
        raise NameError('Not a model available.')

    if args.isTrain:
        os.makedirs(f'./models/{args.model_name}', exist_ok=True)

        train_dataset = load_data(f"{args.train_dir}/train.tsv")
        train_label = train_dataset['label'].values

        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NameError('Not a optimizer available.')
        '''
        elif args.optim == 'adamp':
            optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
        '''


        if args.loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            raise NameError('Not a loss function available.')

        seed_everything(args.seed)
        trainloader, validloader = get_trainLoader(train_dataset, train_label, args.val_ratio, tokenizer)
        train(model, optimizer, criterion, trainloader, validloader, args.epochs, args.model_name, wandb)

    else:

        test_main(model, tokenizer, args.model_name, str(args.chpkt_idx), args.test_dir)
