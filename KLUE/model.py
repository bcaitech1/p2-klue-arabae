import torch
import torch.nn as nn


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=42, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        _, out = self.bert(input_ids=token_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        if self.dr_rate:
            out = self.dropout(out)

        return self.classifier(out)
