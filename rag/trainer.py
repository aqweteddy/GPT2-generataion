import pytorch_lightning as pl
import torch.optim as optim
import torch
from transformers import RagSequenceForGeneration, RagRetriever
import numpy as np

class RagTrainer(pl.LightningModule):
    def __init__(self, **args):
        super(RagTrainer, self).__init__()
        self.save_hyperparameters()
        self.rag_retriever = RagRetriever.from_pretrained(self.hparams['rag_ckpt_path'],
                                                          index_name='custom',
                                                          passages_path=self.hparams['wiki_ds_path'],
                                                          index_path=self.hparams['wiki_index_path'])

        self.rag = RagSequenceForGeneration.from_pretrained(
            self.hparams['rag_ckpt_path'], retriever=self.rag_retriever)

    def generate(self, inputs_ids, attn_mask, **kwargs):
        inputs_ids = inputs_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)

        with torch.no_grad():
            return self.rag.generate(input_ids=inputs_ids, bos_token_id=101, 
                                    attention_mask=attn_mask,
                                     min_length=200, eos_token_id=102,
                                     pad_token_id=0,
                                     **kwargs).detach().cpu().tolist()

    def forward(self, inputs):
        with torch.no_grad():
            return self.rag(**inputs)

    def training_step(self, inputs, batch_idx):
        title, text = inputs
        # pooler_output = self.rag.question_encoder(
        #     title['input_ids'].squeeze(1))[1]
        # docs_dict = self.rag_retriever(
        #     title['input_ids'].squeeze(1).cpu().tolist(), pooler_output.detach().cpu().numpy(), return_tensors="pt")
        # doc_scores = torch.bmm(pooler_output.unsqueeze(
        #     1), docs_dict["retrieved_doc_embeds"].to('cuda').float().transpose(1, 2)).squeeze(1)
        # outputs = self.rag(context_input_ids=docs_dict["context_input_ids"].to('cuda'),
        #                    context_attention_mask=docs_dict["context_attention_mask"].to('cuda'),
        #                    doc_scores=doc_scores,
        #                    decoder_input_ids=text["input_ids"].squeeze(1),
        #                    labels=text['input_ids'].squeeze(1))

        loss = self.rag(input_ids=title['input_ids'].squeeze(1),
                        attention_mask=title['attention_mask'].squeeze(1),
                        labels=text['input_ids'].squeeze(1)
                        ).loss
        return {'loss': loss.mean()}

    def training_epoch_end(self, outputs):
        log = {'mean_loss': torch.stack(
            [x['loss'] for x in outputs]).reshape(-1).mean()}
        self.log_dict(log, prog_bar=True)

    def configure_optimizers(self):
        opt = optim.Adam(self.rag.parameters(), lr=self.hparams['lr'])
        return opt

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--wiki_ds_path', type=str)
        parser.add_argument('--wiki_index_path', type=str)
        parser.add_argument('--rag_ckpt_path', type=str)

        return parser
