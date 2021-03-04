import os
import time, random
from argparse import ArgumentParser
from typing import Counter, List

from tqdm import tqdm
from transformers import BertTokenizerFast

from bert2bert import BERT2BERTTrainer
from gpt2 import GPT2Trainer
from rag import RagTrainer


class Generator:
    def __init__(self, ckpt, model_type, device='cuda'):
        self.model_type = model_type
        if model_type == 'gpt2':
            self.model = GPT2Trainer.load_from_checkpoint(ckpt).to(device)
        elif 'bert2bert' in model_type:
            self.model = BERT2BERTTrainer.load_from_checkpoint(ckpt).to(device)
        elif model_type == 'rag':
            self.model = RagTrainer.load_from_checkpoint(ckpt).to(device)
        print(self.model.device)
        self.model.eval()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    def geneate(self, prompt, maxlen, num_seq, **kwargs):
        encoded_input = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        inputs = encoded_input['input_ids']
        attn_mask = encoded_input['attention_mask']
        result_idx = self.model.generate(inputs,
                                         attn_mask,
                                         max_length=maxlen,
                                         num_beams=15,
                                         num_return_sequences=num_seq,
                                         repetition_penalty=1.5,
                                         temperature=1.5,
                                         do_sample=True,
                                         no_repeat_ngram_size=4,
                                         **kwargs
                                         )
        
        result = self.tokenizer.batch_decode(
            result_idx, skip_special_tokens=True)
        result = [r.replace(' ', '') for r in result]
        return result

def check_kw_in_sent(kws, sent:str):
    tot_cnt = 0
    for kw in kws.replace(' ', ''):
        cnt = sent.count(kw)
        tot_cnt += 0 if cnt == 1 else 1
    return tot_cnt

def generate_bert2bert_nsp(generator, prompt, maxlen, num_seq=3, num_para=3):
    result = [prompt]
    for i in tqdm(range(num_para)):
        new_result = []
        for r in result:
            tmp: List[str] = generator.geneate(r, maxlen, num_seq if i != 0 else 10)
            new_result.extend([r + ' ' + t for t in tmp])
        if i == 0:
            new_result = [r for r in new_result if check_kw_in_sent(prompt, r) >= 2]
            print('\n'.join(new_result))
        if len(new_result) != 0:
            result = random.sample(new_result, min(len(new_result), num_seq))
        else:
            print('no keyword found!! re-generate.')
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--maxlen', type=int, default=200)
    parser.add_argument('--num_seq', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--to', type=str, default='result.txt')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_type', type=str, default='gpt2')


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    generator = Generator(args.ckpt,  args.model_type, args.device)

    start = time.time()
    if args.model_type == 'bert2bert_nsp':
        result = generate_bert2bert_nsp(generator, args.prompt, args.maxlen, args.num_seq)
    else:
        result = generator.geneate(args.prompt, args.maxlen, args.num_seq)
    print(f'{time.time() - start:.3f}')
    with open(args.to, 'w') as f:
        f.writelines(['\n'.join(result)])
