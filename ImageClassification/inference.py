import numpy as np

import torch

from sklearn.metrics import accuracy_score, f1_score


def validation(model, criterion, validloader):
    ###### Validation
    valid_batch_ = len(validloader)

    model.eval()
    valid_perform = np.zeros(3)

    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.float().cuda(), labels.cuda()

            output = model(images)
            loss = criterion(output, labels)

            predict = output.argmax(dim=-1)
            predict = predict.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            valid_acc = accuracy_score(labels, predict)
            valid_f1 = f1_score(labels, predict, average='macro')

            valid_perform += np.array([loss.item(), valid_acc, valid_f1])

            print(f">>>> Validation loss: {valid_perform[0] / valid_batch_:#.5f}, "
                  f"Acc: {valid_perform[1] / valid_batch_:#.2f}, "
                  f"f1 score: {valid_perform[2] / valid_batch_:#.2f}")
            print()

    model.train()
