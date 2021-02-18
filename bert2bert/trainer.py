import pytorch_lightning as pl
import torch.optim as optim
import torch
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizerFast

from bert2bert.model import KeywordsLoss

class BERT2BERTTrainer(pl.LightningModule):
    def __init__(self, lr, **args):
        super(BERT2BERTTrainer, self).__init__()
        self.save_hyperparameters()

        encoder = BertGenerationEncoder.from_pretrained("ckiplab/bert-base-chinese",
                                                        bos_token_id=101,
                                                        eos_token_id=102)
        decoder = BertGenerationDecoder.from_pretrained("ckiplab/bert-base-chinese",
                                                        add_cross_attention=True,
                                                        is_decoder=True,
                                                        bos_token_id=101,
                                                        eos_token_id=102)

        self.bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        if args['with_keywords_loss']:
            self.loss_fct2 = KeywordsLoss(alpha=args['keywords_loss_alpha'])

    def generate(self, inputs_ids, attention_mask=None, **kwargs):
        inputs_ids = inputs_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            return self.bert2bert.generate(input_ids=inputs_ids,
                                           attention_mask=attention_mask,
                                           bos_token_id=101,
                                           min_length=100,
                                           eos_token_id=102,
                                           pad_token_id=0,
                                           **kwargs).detach().cpu().tolist()

    def forward(self, inputs):
        with torch.no_grad():
            return self.bert2bert(**inputs)

    def training_step(self, inputs, batch_idx):
        title, body = inputs
        ret = self.bert2bert(input_ids=title['input_ids'].squeeze(1),
                              attention_mask=title['attention_mask'].squeeze(
                                  1),
                              decoder_input_ids=body['input_ids'].squeeze(1),
                              decoder_attention_mask=body['attention_mask'].squeeze(
                                  1),
                              labels=body['input_ids'].squeeze(1)
                              )
        loss2 = self.loss_fct2(ret.logits, title['input_ids'].squeeze(1)) if self.hparams['with_keywords_loss'] else 0.
        self.log('keyword_loss', loss2, prog_bar=True)
        self.log('clm_loss', ret.loss, prog_bar=True)

        return {'loss': ret.loss + loss2, 'keyword_loss': loss2}

    def training_epoch_end(self, outputs):
        mean_loss = torch.stack(
            [x['loss'] for x in outputs]).reshape(-1).mean()
        self.log('mean_loss', mean_loss)
        # output text
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        result = self.generate(tokenizer('英國 肺炎 台灣', return_tensors='pt')['input_ids'][:, :-1], 
                                         num_beams=10,
                                         num_return_sequences=5,
                                         repetition_penalty=1.3,
                                         temperature=2.,
                                         do_sample=True,
                                         no_repeat_ngram_size=5)
        result = tokenizer.batch_decode(result, skip_special_tokens=True)
        result = [r.replace(' ', '') for r in result]
        newline = '\n'
        self.logger.experiment.log_text(f"英國 肺炎 台灣: {newline.join(result)}")

    def configure_optimizers(self):
        opt = optim.AdamW(self.bert2bert.parameters(), lr=self.hparams['lr'])
        return opt

    @staticmethod
    def add_parser_args(parser):
        # parser.add_argument('--lr', type=float)
        parser.add_argument('--with_keywords_loss', action='store_true')
        parser.add_argument('--keywords_loss_alpha', type=float, default=0.7, help='float > 0.5')
        parser.add_argument('--keywords_loss_fct', type=str, default='kldiv', help='kldiv or mse')

        return parser
