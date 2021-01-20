import pytorch_lightning as pl
from torch.utils import data
from gpt2 import GPT2NewsDataset
from bert2bert import BERT2BERTNewsDataset


class NewsDataModule(pl.LightningDataModule):
    def __init__(self, args, model_type: str):
        super(NewsDataModule, self).__init__()
        if model_type == 'gpt2':
            self.ds = GPT2NewsDataset(args.train_data, args.max_len)
        elif model_type == 'bert2bert':
            self.ds = BERT2BERTNewsDataset(args.train_data, args.title_max_len, args.max_len)
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
        # for bert2bert
        parser.add_argument('--title_max_len', type=int, default=15)
        return parser


if __name__ == '__main__':
    from tqdm import tqdm
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = NewsDataModule.add_parser_args(parser)
    dm = NewsDataModule(parser.parse_args(), 'bert2bert')
    for d in tqdm(dm.train_dataloader()):
        print(d[0])
        print(d[1])
        break
