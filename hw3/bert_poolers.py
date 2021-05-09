import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, BertModel,
    BertEmbeddings, BertEncoder, BertForSequenceClassification, BertPooler,
)


class MeanMaxTokensBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linearTransform = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, hidden_states, *args, **kwargs):
        tmp = torch.sum(hidden_states,dim=1)
        szs = torch.norm(hidden_states, dim=2)
        idx = torch.argmax(szs, dim=1)
        tmp2 =torch.Tensor()
        tmp2 = tmp2.to('cuda')
        
        for i,val in enumerate(idx):
          t = hidden_states[i,val,:]
          tmp2 = torch.cat((tmp2,t.view(-1, len(t))))
        tmp = torch.cat((tmp, tmp2),dim=1)
        
        return self.tanh(self.linearTransform(tmp))


class MyBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

    def softmax_windows(self, x):
        x_strided = x.view(x.shape[0], -1, 2)
        x_softmax_windows = nn.functional.softmax(x_strided, dim=-1)
        x_strided_softmax = x_softmax_windows.view(x_softmax_windows.shape[0], -1)

        return x_strided_softmax

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states[:, 0]
        pooled_output = self.softmax_windows(x)* x
        pooled_output = nn.functional.relu(pooled_output)
        return pooled_output


class MyBertConfig(BertConfig):
    def __init__(self, pooling_layer_type="CLS", **kwargs):
        super().__init__(**kwargs)
        self.pooling_layer_type = pooling_layer_type


class MyBertModel(BertModel):

    def __init__(self, config: MyBertConfig):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        if config.pooling_layer_type == "CLS":
            # See src/transformers/models/bert/modeling_bert.py#L610
            # at huggingface/transformers (9f43a425fe89cfc0e9b9aa7abd7dd44bcaccd79a)
            self.pooler = BertPooler(config)
        elif config.pooling_layer_type == "MEAN_MAX":
            self.pooler = MeanMaxTokensBertPooler(config)
        elif config.pooling_layer_type == "MINE":
            self.pooler = MyBertPooler(config)
        else:
            raise ValueError(f"Wrong pooling_layer_type: {config.pooling_layer_type}")

        self.init_weights()

    @property
    def pooling_layer_type(self):
        return self.config.pooling_layer_type


class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if self.bert.pooling_layer_type in ["CLS", "MEAN_MAX", "MINE"]:
            return super().forward(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states, return_dict
            )
        else:
            raise NotImplementedError