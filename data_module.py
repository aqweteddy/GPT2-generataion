import pytorch_lightning as pl
from torch.utils import data
from transformers import BertTokenizerFast
import json


class NewsDataset(data.Dataset):
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
        return [d['title'] + ' ' + d['body'] for d in data]
        # return [d['body'] for d in data]

    def __getitem__(self, index: int):
        sent = self.recs[index]
        sent = sent.replace(' ', '')
        sent_dct = self.tokenizer(sent, return_tensors='pt',
                                  max_length=self.length,
                                  padding='max_length',
                                  truncation=True)
        # print(sent_dct)
        return (sent_dct['input_ids'], sent_dct['token_type_ids'], sent_dct['attention_mask'])


class NewsDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(NewsDataModule, self).__init__()
        self.ds = NewsDataset(args.train_data, args.max_len)
        self.args = args

    def train_dataloader(self):
        return data.DataLoader(self.ds, batch_size=self.args.batch_size,
                               shuffle=True, num_workers=self.args.num_workers)

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--train_data', type=str)
        parser.add_argument('--max_len', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--num_workers', type=int, default=10)
        return parser


if __name__ == '__main__':
    from tqdm import tqdm
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = NewsDataModule.add_parser_args(parser)
    dm = NewsDataModule(parser.parse_args())
    for d in tqdm(dm.train_dataloader()):
        print(d[0])
        break
