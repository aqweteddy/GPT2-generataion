import os, comet_ml
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from pytorch_lightning.callbacks import ModelCheckpoint

from gpt2 import GPT2Trainer
from bert2bert import BERT2BERTTrainer
from rag import RagTrainer
from data_module import NewsDataModule


parser = ArgumentParser()
parser.add_argument('--model_type', type=str, default='gpt2')
parser.add_argument('--save_top_k', type=int, default=4)
parser.add_argument('--exp_name', type=str, default='exp1')
parser.add_argument('--gpuid', type=str, default='0')

parser = NewsDataModule.add_parser_args(parser)
parser = GPT2Trainer.add_parser_args(parser)
parser = BERT2BERTTrainer.add_parser_args(parser)
parser = RagTrainer.add_parser_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

checkpoint_callback = ModelCheckpoint(monitor='mean_loss',
                                      save_top_k=4,
                                      verbose=True,
                                      mode='min')
train_dm = NewsDataModule(args, args.model_type)

if args.model_type == 'gpt2':
    model = GPT2Trainer(**vars(args))
elif args.model_type == 'bert2bert' or args.model_type == 'bert2bert_nsp':
    model = BERT2BERTTrainer(**vars(args))
elif args.model_type == 'rag':
    model = RagTrainer(**vars(args))

# model.bert2bert.save_pretrained('rag_ckpt')
logger = CometLogger(project_name='kw2text', experiment_name=args.exp_name)
trainer = pl.Trainer.from_argparse_args(args,
                                        logger=logger,
                                        reload_dataloaders_every_epoch=True,
                                        callbacks=[checkpoint_callback])
logger.finalize('finished')
trainer.fit(model, train_dm)

