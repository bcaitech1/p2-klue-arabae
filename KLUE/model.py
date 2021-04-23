import torch
import torch.nn as nn

class kobert_Classifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=42, dr_rate=0.0):
        super(kobert_Classifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        out = self.bert(input_ids=token_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
		
		if self.dr_rate:
			out = self.dropout(out)

        return self.classifier(out)
    
class koelectra_Classifier(nn.Module):
    def __init__(self, electra, hidden_size=768, num_classes=42, dr_rate=0.0):
        super(koelectra_Classifier, self).__init__()
        self.electra = electra
        self.dr_rate = dr_rate
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        torch.nn.init.xavier_uniform_(self.pooler.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        out = self.electra(input_ids=token_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
        
        out = out[:, 0, :] # take <s> token (equiv. to [CLS])
        out = self.pooler(out)
        out = torch.nn.functional.gelu(out)  # although BERT uses tanh here, it seems Electra authors used gelu here
        if self.dr_rate:
			out = self.dropout(out)

        return self.classifier(out)
		
class roberta_Classifier(nn.Module):
    def __init__(self, roberta, hidden_size=1024, num_classes=42, dr_rate=0.0):
        super(roberta_Classifier, self).__init__()
        self.roberta = roberta
        self.dr_rate = dr_rate
        
        self.pooler = nn.Linear(hidden_size, hidden_size//2)
        self.classifier = nn.Linear(hidden_size//2, num_classes)
        torch.nn.init.xavier_uniform_(self.pooler.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        
        if self.dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids=None):
        out = self.roberta(input_ids=token_ids, attention_mask=attention_mask)[0]
        
        out = out[:, 0, :] # take <s> token (equiv. to [CLS])
        out = self.pooler(out)
        out = torch.nn.functional.gelu(out)
        
        if self.dr_rate:
            out = self.dropout(out)

        return self.classifier(out)
    
'''
Reference:
    https://github.com/monologg/R-BERT
'''
class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

    
class r_roberta_Classifier(nn.Module):
    def __init__(self, roberta, hidden_size=1024, num_classes=42, dr_rate=0.0):
        super(r_roberta_Classifier, self).__init__()
        self.roberta = roberta
        self.dr_rate = dr_rate
        
        self.cls_fc = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.entity_fc = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.label_classifier = FCLayer(hidden_size//2 * 3, num_classes, self.dr_rate, False)

    def forward(self, token_ids, attention_mask, segment_ids=None):
        out = self.roberta(input_ids=token_ids, attention_mask=attention_mask)[0]
        
        entity_end_position = torch.where(token_ids == 2)[1]
        entity1_end, entity2_end = entity_end_position[0], entity_end_position[2]
        
        cls_vector = out[:, 0, :] # take <s> token (equiv. to [CLS])
        entity1_vector = out[:, 1:entity1_end, :] # Get Entity vector
        entity2_vector = out[:, entity1_end+2:entity2_end, :]
        
        entity1_vector = torch.mean(entity1_vector, dim=1) # Average
        entity2_vector = torch.mean(entity2_vector, dim=1)
        
        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        cls_embedding = self.cls_fc(cls_vector)
        e1_embedding = self.entity_fc(entity1_vector)
        e2_embedding = self.entity_fc(entity2_vector)
        
        # Concat -> fc_layer
        concat_embedding = torch.cat([cls_embedding, e1_embedding, e2_embedding], dim=-1)
        
        return self.label_classifier(concat_embedding)


def get_tokenizer(args):
	if args.model == 'kobert':
		tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
		
	elif args.model == 'multi':
		tokenizer = AutoTokenizer.from_pretrained("sangrimlee/bert-base-multilingual-cased-korquad")
		
	elif args.models == 'koelectra':
		tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
		
    elif args.model == 'r_roberta':
		tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
		
    elif args.model == 'roberta':
		tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
	
	else:
		raise NotImplementedError('Tokenizer & Model not available')
	
	return tokenizer

def get_model(args):
	if args.model == 'kobert':
		feature_model = BertModel.from_pretrained("monologg/kobert")
		model = kobert_Classifier(feature_model, dr_rate=args.dp)
		
	elif args.model == 'multi':
		feature_model = BertModel.from_pretrained("sangrimlee/bert-base-multilingual-cased-korquad")
		model = kobert_Classifier(feature_model, dr_rate=args.dp)
		
	elif args.models == 'koelectra':
		feature_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
		model = koelectra_Classifier(feature_model, dr_rate=args.dp)
		
    elif args.model == 'r_roberta':
		feature_model = RobertaModel.from_pretrained("xlm-roberta-large", add_pooling_layer=False)
           model = r_roberta_Classifier(feature_model, dr_rate=args.dp)
		
    elif args.model == 'roberta':
		feature_model = RobertaModel.from_pretrained("xlm-roberta-large", add_pooling_layer=False)
		model = roberta_Classifier(feature_model)
	
	else:
		raise NotImplementedError('Tokenizer & Model not available')
	
	return model