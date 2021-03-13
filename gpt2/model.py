import torch
from transformers import BertTokenizerFast, GPT2LMHeadModel


class GPT2LMHeadWithCalibration(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2LMHeadWithCalibration, self).__init__(config)

    def adjust_logits_during_generation(self, logits,  cur_len, max_length):
        """
        logits [sent_len, vocab_size]
        """

        if not hasattr(self, 'first_adjust'):
            logits = torch.softmax(logits, -1)
            logits[-1, :] = self.calibrate(logits[-1, :])
            self.first_adjust = True

        return logits

    def calibrate(self, logits: torch.tensor):
        """calibration

        Args:
            logits (torch.tensor): [seq_len, vocab_size]

        Returns:
            [type]: [description]
        """
        if not hasattr(self, 'calibrate_dct'):
            print('calculate calibration weight...')
            self.calibrate_dct = self.__get_calibrate_attr()

        q_head = torch.matmul(logits, self.calibrate_dct['W'])\
            + self.calibrate_dct['b']
        q_head = torch.softmax(q_head, -1)
        return q_head

    @torch.no_grad()
    def __get_calibrate_attr(self):
        inp_id = torch.tensor([[101, 103], [101, 100]]).to(self.device)
        p_cf = self(inp_id).logits[:, -1, :]
        p_cf, _ = p_cf.min(0)
        p_cf = torch.softmax(p_cf, -1)
        w = torch.diag(p_cf)
        b = -1 * p_cf
        return {'W': torch.zeros_like(w), 'b': b}


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    inp_id = torch.tensor(tokenizer.encode('以我自己')[:-1]).unsqueeze(0)
    gpt2 = GPT2LMHeadWithCalibration.from_pretrained(
        'ckiplab/gpt2-base-chinese')
    torch.manual_seed(10)

    result = gpt2.generate(input_ids=inp_id, bos_token_id=101, max_length=200,
                           min_length=10, eos_token_id=102,
                           pad_token_id=0,  num_beams=20, do_sample=True,
                           repetition_penalty=1.5,
                           temperature=1.0,
                           no_repeat_ngram_size=4,
                           ).detach().cpu().tolist()
    print(tokenizer.decode(result[0]))
