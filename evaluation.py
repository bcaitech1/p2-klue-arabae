import torch
import numpy as np
import pandas as pd

from model import *
from dataloader import *
from torch.utils.data import DataLoader


def load_test_dataset(args, tokenizer):
    if 'ner' in args.test_file:
        test_dataset = ner_load_data(f"{args.test_dir}/{args.test_file}.tsv")
    else:
        test_dataset = load_data(f"{args.test_dir}/{args.test_file}.tsv")
	test_label = test_dataset['label'].values
    
	# tokenizing dataset
	entity_between = '</s></s>' if args.model == 'r_roberta' or args.model == 'roberta' else '[SEP]'
    
    tokenized_test = tokenized_dataset(test_dataset, entity_between, tokenizer)
    return tokenized_test, test_label


def test_eval(model, tokenized_sent):
    testloader = DataLoader(tokenized_sent,
                            shuffle=False)
    model.eval()
    output_pred = []
    for i, data in enumerate(testloader):
        with torch.no_grad():
            logits = model(
                data['input_ids'].cuda(),
                data['attention_mask'].cuda(),
                data['token_type_ids'].cuda()
            )
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.extend(list(result))
    return list(np.array(output_pred).reshape(-1))


def kfold_test_eval(args, tokenized_sent):
    testloader = DataLoader(tokenized_sent,
                            shuffle=False)
    
    fold_logits = np.zeros((1000, 42))
    for fold in range(1, 9):
        # load my model
        model = get_model(args)
        load_path = f'./models/{args.model_name}/{fold}-fold/best.pt'
        
        if os.path.isfile(load_path):
            model.load_state_dict(torch.load(load_path))
            model.cuda()

            model.eval()
            output_pred = []

            for i, data in enumerate(testloader):
                with torch.no_grad():
                    logits = model(
                        data['input_ids'].cuda(),
                        data['attention_mask'].cuda()
                    )
                fold_logits[i] += logits.squeeze(0).detach().cpu().numpy()
    np.save(f'./results/{args.model_name}_logits.npy', fold_logits)
    
    result = np.argmax(fold_logits, axis=1)
    output_pred = list(result)
    
    return list(np.array(output_pred).reshape(-1))
	

def test_main(args, model, tokenizer):
    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """

    # load test datset
    test_dataset, test_label = load_test_dataset(args, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)

    # predict answer
    pred_answer = test_eval(args, model, test_dataset)

    # make csv file with predicted answer
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv(f'./results/{args.model_name}-{args.chkpt_idx}.csv', index=False)
	
	

if __name__ == '__main__':
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--model', type=str, default='r_roberta', help='model type (kobert, koelectra, multi, roberta, r_roberta; default)')
    parser.add_argument('--test_dir', type=str, default='../input/data/test')
    parser.add_argument('--test_file', type=str, default='test', help='choose test; default, ner_test')
    
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--chkpt_idx', type=int, default=10, help='checkpoint of models for submit')
    
    
    args = parser.parse_args()
    test_main(args)
