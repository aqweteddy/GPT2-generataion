
from torch.utils import data
from transformers import BertTokenizerFast
import json

class RagNewsDataset(data.Dataset):
    def __init__(self, train_data, title_maxlen, maxlen):
        self.titles, self.bodies = self.load_file(train_data)
        self.tokenizer = BertTokenizerFast.from_pretrained('ckiplab/albert-base-chinese')
        self.body_length = maxlen
        self.title_length = title_maxlen

    def __len__(self) -> int:
        return len(self.titles)

    @staticmethod
    def load_file(file):
        with open(file) as f:
            data = json.load(f)
        body, title = [], []
        for d in data:
            for b in d['body']:
                body.append(b)
                title.append(d['title'])
        return title, body

    def __getitem__(self, index: int):
        title = self.titles[index]
        body = self.bodies[index]
        title = title.replace('/', '')
        if '】' in body:
            body = body[body.index('】'):]
        title_idx = self.tokenizer(title, return_tensors='pt',
                                  max_length=self.title_length,
                                  padding='max_length',
                                  add_special_tokens=False,
                                  truncation=True)
        body_idx = self.tokenizer(body, return_tensors='pt',
                                  max_length=self.body_length,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        return title_idx, body_idx