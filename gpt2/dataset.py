from torch.utils import data
from transformers import BertTokenizerFast
import json, re


class GPT2NewsDataset(data.Dataset):
    def __init__(self, file, length=500) -> None:
        self.recs = self.load_file(file)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.length = length

    def __len__(self) -> int:
        return len(self.recs)

    @staticmethod
    def load_file(file):
        with open(file) as f:
            data = json.load(f)
        body = []
        chinese = '[\u4e00-\u9fa5]+'

        for d in data:
            d['title'] = re.findall(chinese, d['title'])
            d['title'] = ''.join(d['title'])
            if len(d['title']) > 5:
                b = ''.join(d['body'])
                body.extend([d['title'] + ':' + b])
        return body

    def __getitem__(self, index: int):
        sent = self.recs[index]
        sent = sent.replace(' ', '')
        sent_dct = self.tokenizer(sent, return_tensors='pt',
                                  max_length=self.length,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        # print(sent_dct)
        return (sent_dct['input_ids'], sent_dct['token_type_ids'],
                sent_dct['attention_mask'])


if __name__ == '__main__':
    news_ds = GPT2NewsDataset('../data/all.json', 50)
    print(news_ds[5])
