
from torch.utils import data
from transformers import BertTokenizerFast
import json

class BERT2BERTNewsDataset(data.Dataset):
    def __init__(self, train_data, maxlen):
        self.titles, self.bodies = self.load_file(train_data)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.length = maxlen

    def __len__(self) -> int:
        return len(self.titles)

    @staticmethod
    def load_file(file):
        with open(file) as f:
            data = json.load(f)
        return [d['title'] for d in data], [d['body'] for d in data]

    def __getitem__(self, index: int):
        title = self.titles[index]
        body = self.bodies[index]
        title = title.replace('/', '')
        # title = title.replace('/', '')
        
        title_idx = self.tokenizer(title, return_tensors='pt',
                                  max_length=self.length,
                                  padding='max_length',
                                  add_special_tokens=False,
                                  truncation=True)
        body_idx = self.tokenizer(body, return_tensors='pt',
                                  max_length=self.length,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        return title_idx, body_idx