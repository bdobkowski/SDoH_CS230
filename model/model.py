import transformers
import torch.nn as nn
import torch

class BertPretrained(nn.Module):
    '''
    Transfer learning with Bert pretrained model.
    '''
    def __init__(self, model_name, drop=0.3, num_classes=1, linear_in=768):
        super(BertPretrained, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(drop)
        self.out = nn.Linear(linear_in, num_classes)
        # self.out_activation = nn.Sigmoid()

    def forward(self, ids, masks, token_type_ids):
        out1, out2 = self.model(ids, attention_mask=masks, token_type_ids=token_type_ids).values()
        # TODO: Fix stupid bug above, out1 and out2 are both strings
        # Apparently API call change from huggingface, added .values 
        out2 = self.dropout(out2)
        out2 = self.out(out2)
        # Added sigmoid activation since outputs < 0 we triggering cuda assertion error
        # out2 = self.out_activation(out2)
        return out2