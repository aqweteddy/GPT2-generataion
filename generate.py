from gpt2 import GPT2Trainer
from bert2bert import BERT2BERTTrainer
from transformers import BertTokenizerFast
from argparse import ArgumentParser
import os, time


class Generator:
    def __init__(self, ckpt, model_type, device='cuda'):
        self.model_type = model_type
        if model_type == 'gpt2':
            self.model = GPT2Trainer.load_from_checkpoint(ckpt).to(device)
        elif model_type == 'bert2bert':
            self.model = BERT2BERTTrainer.load_from_checkpoint(ckpt).to(device)
        print(self.model.device)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    def geneate(self, prompt, maxlen, num_seq, **kwargs):
        encoded_input = self.tokenizer(prompt, return_tensors='pt')
        inputs = encoded_input['input_ids'][:, :-1]
        attn_mask = encoded_input['attention_mask'][:, :-1]
        result_idx = self.model.generate(inputs,
                                         attn_mask,
                                         max_length=maxlen,
                                         num_beams=10,
                                         num_return_sequences=num_seq,
                                         repetition_penalty=1.3,
                                        #  num_beam_groups=num_seq,
                                        #  diversity_penalty=1.3,
                                         do_sample=True,
                                         no_repeat_ngram_size=5,
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
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--model_type', type=str, default='gpt2')


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    generator = Generator(args.ckpt,  args.model_type, args.device)

    start = time.time()
    result = generator.geneate(args.prompt, args.maxlen, args.num_seq)
    print(f'{time.time() - start:.3f}')
    with open(args.to, 'w') as f:
        f.writelines(['\n'.join(result)])