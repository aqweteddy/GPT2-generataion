import os
from argparse import ArgumentParser
from gpt2 import GPT2Trainer
from bert2bert import BERT2BERTTrainer
from data_module import NewsDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = ArgumentParser()
parser.add_argument('--model_type', type=str, default='gpt2')
parser.add_argument('--save_top_k', type=int, default=4)
parser = NewsDataModule.add_parser_args(parser)
parser = GPT2Trainer.add_parser_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()


checkpoint_callback = ModelCheckpoint(monitor='mean_loss',
                                      save_top_k=args.save_top_k,
                                      mode='min')
train_dm = NewsDataModule(args, args.model_type)
if args.model_type == 'gpt2':
    model = GPT2Trainer(**vars(args))
elif args.model_type == 'bert2bert':
    model = BERT2BERTTrainer(**vars(args))

trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
trainer.fit(model, train_dm)
