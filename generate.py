from trainer import GPT2Trainer
from transformers import BertTokenizerFast, GPT2LMHeadModel
from argparse import ArgumentParser


class Generator:
    def __init__(self, ckpt, device='cuda'):
        self.model = GPT2Trainer.load_from_checkpoint(ckpt).to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    def geneate(self, prompt, maxlen, num_seq, **kwargs):
        encoded_input = self.tokenizer(prompt, return_tensors='pt')
        inputs = encoded_input['input_ids'][:, :-1]
        attn_mask = encoded_input['attention_mask'][:, :-1]
        result_idx = self.model.generate(inputs,
                                         attention_mask=attn_mask,
                                         max_length=maxlen,
                                         num_beams=10,
                                         num_return_sequences=num_seq,
                                         repetition_penalty=1.3,
                                         do_sample=True,
                                         **kwargs
                                         )
        
        result = self.tokenizer.batch_decode(
            result_idx, skip_special_tokens=True)
        result = [r.replace(' ', '') for r in result]
        return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--maxlen', type=int, default=200)
    parser.add_argument('--num_seq', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--to', type=str, default='result.txt')

    args = parser.parse_args()

    generator = Generator(args.ckpt, args.device)
    result = generator.geneate(args.prompt, args.maxlen, args.num_seq)

    with open(args.to, 'w') as f:
        f.writelines(['\n'.join(result)])