import pandas as pd
import torch
import numpy as np

from dataloader import load_data, tokenized_dataset, RE_Dataset
from torch.utils.data import DataLoader


def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label


def test_eval(model, tokenized_sent):
    testloader = DataLoader(tokenized_sent,
                            batch_size=32,
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


def test_main(model, tokenizer, model_name, chkpt_idx, test_dir):
    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """

    # load my model
    load_path = f'./models/{model_name}/chkpt-{chkpt_idx}.pt'
    model.load_state_dict(torch.load(load_path))
    model.cuda()

    # load test datset
    test_dataset_dir = f"{test_dir}/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)

    # predict answer
    pred_answer = test_eval(model, test_dataset)

    # make csv file with predicted answer
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv(f'./results/{model_name}-{chkpt_idx}.csv', index=False)
