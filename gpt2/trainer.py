import sys

import pytorch_lightning as pl
import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel
from utils import KeywordsLoss

sys.path.append("..")


class GPT2Trainer(pl.LightningModule):
    def __init__(self, lr, **args):
        super(GPT2Trainer, self).__init__()
        self.save_hyperparameters()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            'ckiplab/gpt2-base-chinese')
        if args['with_keywords_loss']:
            self.loss_fct2 = KeywordsLoss(
                alpha=args['keywords_loss_alpha'], loss_fct=args['keywords_loss_fct'])

    def generate(self, inputs_ids, attn_mask, **kwargs):
        inputs_ids = inputs_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)

        with torch.no_grad():
            return self.gpt2.generate(input_ids=inputs_ids, bos_token_id=101, attention_mask=attn_mask,
                                      min_length=200, eos_token_id=102,
                                      pad_token_id=0,
                                      **kwargs).detach().cpu().tolist()

    def forward(self, inputs):
        with torch.no_grad():
            return self.gpt2(**inputs)

    def training_step(self, inputs, batch_idx):
        # inputs = {key: torch.stack(val) for key, val in inputs.items()}
        ret = self.gpt2(input_ids=inputs[0].squeeze(1),
                        token_type_ids=inputs[1].squeeze(1),
                        attention_mask=inputs[2].squeeze(1),
                        labels=inputs[0].squeeze(1))
        loss2 = self.loss_fct2(ret.logits, inputs[0].squeeze(
            1)) if self.hparams['with_keywords_loss'] else 0.
        self.log('keyword_loss', loss2, prog_bar=True)
        self.log('lm_loss', ret.loss, prog_bar=True)
        return {'loss': ret.loss + loss2}

    def training_epoch_end(self, outputs):
        log = {'mean_loss': torch.stack(
            [x['loss'] for x in outputs]).reshape(-1).mean()}
        self.log_dict(log, prog_bar=True)
        self.gpt2.save_pretrained(f'gpt2_ckpt/epoch={self.current_epoch}')

    def configure_optimizers(self):
        opt = optim.Adam(self.gpt2.parameters(), lr=self.hparams['lr'])
        return opt

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lr', type=float)

        # parser.add_argument('--with_keywords_loss', action='store_true')
        # parser.add_argument('--keywords_loss_alpha', type=float, default=0.7, help='float > 0.5')
        # parser.add_argument('--keywords_loss_fct', type=str, default='kldiv', help='kldiv or mse')
        return parser
