import numpy 
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BertDataset(Dataset):
    def __init__(self, text, tokenizer, max_len=512, target=None):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        output = self.tokenizer.encode_plus(
            " ".join(str(self.text[idx]).split()),
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            pad_to_max_length=True
        )

        ids = output['input_ids']
        masks = output['attention_mask']
        token_type_ids = output['token_type_ids']

        if self.target is not None:
            return {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(masks, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.target[idx], dtype=torch.float)
            }
        else:
            return {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(masks, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
            }