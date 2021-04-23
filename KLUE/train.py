import os
import random
import argparse
import numpy as np
import sys

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

from loss import *
from model import *
from dataloader import *
from transformers import AutoTokenizer, BertModel, ElectraModel, RobertaModel

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


def train(args, criterion, wandb):
	tokenizer = get_tokenizer(args)
	
    if 'ner' in args.train_file:
        all_dataset = ner_load_data(f"{args.train_dir}/{args.train_file}.tsv")
    else:
        all_dataset = load_data(f"{args.train_dir}/{args.train_file}.tsv")
    all_label = all_dataset['label'].values
    
    kf = StratifiedKFold(n_splits=8, random_state=42, shuffle=True)
    fold_idx = 1

    for train_index, test_index in kf.split(all_dataset, all_label):
        
        os.makedirs(f'./models/{args.model_name}/{fold_idx}-fold', exist_ok=True)
        ### Model Select
		model = get_model(args)
        model.cuda()
        
		### Optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        train_data, valid_data = all_dataset.iloc[train_index], all_dataset.iloc[test_index]
        train_label, valid_label = all_label[train_index], all_label[test_index]

        trainloader, validloader = get_trainLoader(args, train_data, valid_data, train_label, valid_label, tokenizer)

        total_batch_ = len(trainloader)
        valid_batch_ = len(validloader)
        
        best_val_loss, best_val_acc = np.inf, 0
    
        print(f"---------------------------------- {fold_idx} ----------------------------------")

		for i in range(args.epochs + 1):
			model.train()
			epoch_perform, batch_perform = np.zeros(2), np.zeros(2)

			for j, v in enumerate(trainloader):
				input_ids, attention_mask, labels = v['input_ids'].cuda(), v['attention_mask'].cuda(), v['labels'].cuda()
				
				if args.model == 'roberta' or args.model == 'r_roberta':
					token_type_ids = None
				else:
					token_type_ids = v['token_type_ids'].cuda()
				optimizer.zero_grad()
				
				output = model(input_ids, attention_mask, token_type_ids) ## label을 안 넣어서 logits값만 출력

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
						f"Epoch {i:#04d} #{j + 1:#03d} -- loss: {batch_perform[0] / 50:#.5f}, acc: {batch_perform[1] / 50:#.4f}")
					batch_perform = np.zeros(2)
			
			print(
				f"Epoch {i:#04d} loss: {epoch_perform[0] / total_batch_:#.5f}, acc: {epoch_perform[1] / total_batch_:#.2f}")
			wandb.log({
				"epoch": i,
				"Train epoch Loss": epoch_perform[0] / total_batch_,
				"Train epoch Acc": epoch_perform[1] / total_batch_})
			###### Validation

			model.eval()
			valid_perform = np.zeros(2)
			with torch.no_grad():
				for v in validloader:
					input_ids, attention_mask, valid_labels = v['input_ids'].cuda(), v['attention_mask'].cuda(), v['labels'].cuda()
				
					if args.model == 'roberta' or args.model == 'r_roberta':
						token_type_ids = None
					else:
						token_type_ids = v['token_type_ids'].cuda()
					
					valid_output = model(input_ids, attention_mask, token_type_ids)
					valid_loss = criterion(valid_output, valid_labels)

					valid_predict = valid_output.argmax(dim=-1)
					valid_predict = valid_predict.detach().cpu().numpy()
					valid_labels = valid_labels.detach().cpu().numpy()

					valid_acc = accuracy_score(valid_labels, valid_predict)

					valid_perform += np.array([valid_loss.item(), valid_acc])

			###### Model save
			val_total_loss = valid_perform[0] / valid_batch_
			val_total_acc = valid_perform[1] / valid_batch_
			best_val_loss = min(best_val_loss, val_total_loss)

			if val_total_acc > best_val_acc and val_total_acc >= 0.74:
				print(f"New best model for val accuracy : {val_total_acc:#.4f}! saving the best model..")
				torch.save(model.state_dict(), f"./models/{args.model_name}/{fold_idx}-fold/best.pt")
				best_val_acc = val_total_acc
		
			print(
				f">>>> Validation loss: {val_total_loss:#.5f}, Acc: {val_total_acc:#.4f}")
			print()
			wandb.log({
				"epoch": i,
				"Valid Loss": val_total_loss,
				"Valid Acc": val_total_acc})
				
		fold_idx +=1



if __name__ == '__main__':
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='klue-baseline', help='wandb project name')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    
    parser.add_argument('--model', type=str, default='r_roberta', help='model type (kobert, koelectra, multi, roberta, r_roberta; default)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 5e-5)')
    parser.add_argument('--smoothing', type=float, default=0.5, help='smoothing level (default: 0.5)')
    parser.add_argument('--dp', type=float, default=None, help='Dropout rate of Classifier (default: None)')
    
    parser.add_argument('--train_dir', type=str, default='../input/data/train')
	parser.add_argument('--isAug', type=str2bool, default=False, help='choose Augmentation(true) or Not(false; default)')
    parser.add_argument('--train_dir', type=str, default='../input/data/train')
    parser.add_argument('--train_file', type=str, default='train', help='choose train; default, gold_train, pororo_train, gold_pororo_train, ner_train')
	
    parser.add_argument('--model_name', type=str, required=True)

    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    wandb.init(project=args.project_name)
    wandb.run.name = f'{args.model_name}'
    wandb.config.update(args)
		
	criterion = LabelSmoothingLoss(smoothing=args.smoothing)
	train(args, criterion, wandb)

