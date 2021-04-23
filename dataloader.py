import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# Dataset 구성.
class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {'sentence': dataset[1], 'entity_01': dataset[2], 'entity_02': dataset[5], 'label': label,})
    return out_dataset


# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    with open('../input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset


# ner tag가 붙은 tsv 파일을 불러옵니다.
def ner_load_data(dataset_dir):
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    dataset = pd.DataFrame(
        {'sentence':dataset[0], 'entity_01': dataset[1], 'entity_02': dataset[2], 'label': dataset[3]})
    return dataset


# bert input을 위한 tokenizing.
def tokenized_dataset(dataset, entity_between, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + entity_between + e02
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=150,
        add_special_tokens=True
    )
    return tokenized_sentences


def get_trainLoader(args, train_data, valid_data, train_label, valid_label, tokenizer):
    # tokenizing dataset
    if args.isAug:
        train_num = len(train_data)
        train_pieces = dict(list(train_data.groupby('label')))
        
        aug_dataset = load_data("../input/data/aug/aug1.tsv")
        pieces = dict(list(aug_dataset.groupby('label')))
        
        for i in pieces.keys():
            df_shuffled = pieces[i].sample(frac=len(train_pieces[i])/train_num).reset_index(drop=True)
            train_data = pd.concat([train_data, df_shuffled])
    
    # remove under 8 samples
    '''
    del_labels = []
    train_pieces = dict(list(train_data.groupby('label')))
    for i in train_pieces.keys():
        if len(train_pieces[i]) < 8:
            del_labels.append(i)
    train_data[~train_data['label'].isin(del_labels)]
    '''
    entity_between = '</s></s>' if args.model == 'r_roberta' or args.model == 'roberta' else '[SEP]'
    tokenized_train = tokenized_dataset(train_data, entity_between, tokenizer)
    tokenized_valid = tokenized_dataset(valid_data, entity_between, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    trainloader = DataLoader(RE_train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=4
                             )

    validloader = DataLoader(RE_valid_dataset,
                             shuffle=False,
                             num_workers=4
                             )

    return trainloader, validloader