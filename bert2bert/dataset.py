
from torch.utils import data
from transformers import BertTokenizerFast
import json, torch
import random
from tqdm import tqdm

class BERT2BERTNewsDataset(data.Dataset):
    def __init__(self, train_data, sent1_maxlen, maxlen):
        self.sent1s, self.sent2s = self.load_file(train_data)
        self.tokenizer = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese')
        self.sent2_length = maxlen
        self.sent1_length = sent1_maxlen
        print(f'data_len:{len(self.sent1s)}')

    def __len__(self) -> int:
        return len(self.sent1s)
 
    @staticmethod
    def check_kw_in_sent(kws, sent):
        kws = kws.split(',')
        result = [kw for kw in kws if kw in sent]
        return result

    def load_file(self, file):
        with open(file) as f:
            data = json.load(f)
        sent2, sent1 = [], []
        for d in tqdm(data):
            # r = self.check_kw_in_sent(d['keywords'], d['title'])
            # sent2.append(d['title'])
            # sent2.append(''.join([char for char in d['title'] if not char.isdigit() and not char.isalpha() and char not in '. ,']))
            # sent1.append(r)
            for b in d['body']:
                r = self.check_kw_in_sent(d['keywords'], b)
                if len(r) > 1:
                    sent2.append(b)
                    sent1.append(r)
        return sent1, sent2

    def __getitem__(self, index: int):
        sent1 = self.sent1s[index]
        sent2 = self.sent2s[index]
        # print(sent1, sent2)
        # kws = sent1.split(',')
        #if len(kws) > 3:
        #    sent1 = ','.join(random.sample(sent1.split(','), 3))
        #else:
        #    sent1 = ','.join(kws)
        # kws = list(filter(lambda x: True if sent2.find(x) != -1 else False, kws))
        if len(sent1) > 4:
            sent1 = random.sample(sent1, 3)
        sent1 = ','.join(sent1)
        sent1_idx = self.tokenizer(sent1, return_tensors='pt',
                                  max_length=self.sent1_length,
                                  padding='max_length',
                                  add_special_tokens=False,
                                  truncation=True)
        sent2_idx = self.tokenizer(sent2, return_tensors='pt',
                                  max_length=self.sent2_length,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        
        return sent1_idx, sent2_idx


class BERT2BERTNspDataset(data.IterableDataset):
    def __init__(self, train_data, sent1_maxlen, maxlen):
        self.sent1s, self.sent2s = self.load_file(train_data)
        print(f'data_len: {len(self.sent1s)}')
        self.tokenizer = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese')
        self.sent2_length = maxlen
        self.sent1_length = sent1_maxlen
    
    def load_file(self, file):
        with open(file) as f:
            data = json.load(f)
        sents2, sents1 = [], []
        for d in random.sample(data, 40000):
            sent1 = [d['keywords']]
            if len(d['body']) == 0:
                continue
            sent_idx = 0
            sent2 = [d['body'][sent_idx]]
            while BERT2BERTNewsDataset.check_kw_in_sent(d['keywords'], sent2) > 1 and sent_idx < len(d['body']):
                sent_idx += 1
                sent2 = [d['body'][sent_idx]]
            if sent_idx >= len(d['body']):
                continue
            for b1, b2 in zip(d['body'][sent_idx], d['body'][sent_idx + 1:]):
                sent1.append(b1)
                sent2.append(b2)
            sents1.append(sent1)
            sents2.append(sent2)
                
        return sents1, sents2
    
    def tokenize_sents_pair(self, sent1, sent2):
        sent1_idx = self.tokenizer(sent1, return_tensors='pt',
                                  max_length=self.sent1_length,
                                  padding='max_length',
                                  add_special_tokens=False,
                                  truncation=True)
        sent2_idx = self.tokenizer(sent2, return_tensors='pt',
                                  max_length=self.sent2_length,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        return sent1_idx, sent2_idx

    def per_proc(self, start, end):
        for sents1, sents2 in zip(self.sent1s[start:end], self.sent2s[start:end]):
            now_text_list = []
            for sent1, sent2 in zip(sents1, sents2):
                now_text_list.append(sent1)
                now_text = ' '.join(now_text_list)
                
                while len(now_text) > self.sent1_length:
                    now_text_list = [now_text_list[0]] + now_text_list[2:]
                    now_text = ' '.join(now_text_list)
                # print(now_text, sent2)
                sent1_idx, sent2_idx = self.tokenize_sents_pair(now_text, sent2)
                yield sent1_idx, sent2_idx

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = int(len(self.sent1s) / float(worker_info.num_workers))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.sent1s))
            return iter(self.per_proc(iter_start, iter_end))
        else:
            return self.per_proc(0, len(self.sent1s))
