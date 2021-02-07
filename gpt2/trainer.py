import pytorch_lightning as pl
import torch.optim as optim
import torch
from transformers import GPT2LMHeadModel


class GPT2Trainer(pl.LightningModule):
    def __init__(self, lr, **args):
        super(GPT2Trainer, self).__init__()
        self.save_hyperparameters()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            'ckiplab/gpt2-base-chinese')

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
        outputs = self.gpt2(input_ids=inputs[0].squeeze(1),
                            token_type_ids=inputs[1].squeeze(1),
                            attention_mask=inputs[2].squeeze(1),
                            labels=inputs[0].squeeze(1))
        return {'loss': outputs[0]}
    
    def training_epoch_end(self, outputs):
        log = {'mean_loss': torch.stack([x['loss'] for x in outputs]).reshape(-1).mean() }
        self.log_dict(log, prog_bar=True)
        
    def configure_optimizers(self):
        opt = optim.Adam(self.gpt2.parameters(), lr=self.hparams['lr'])
        return opt

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lr', type=float)
        return parser
