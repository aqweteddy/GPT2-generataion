# README

* based on `pytorch` / `transformers` and `pytorch_lightning`

## train data description

* json
```json
[
    {"title": "xxxx", "body": "xxxxx"},
    {"title": "xxxx", "body": "xxxxx"}
    ...
]
```

## GPT2

* pretrained model: CKIP Lab GPT2-base
* Two generation methods:
    * contextual calibration
    * normal beam search

### fine-tune

```sh
python train_script.py --model_type gpt2 --train_data data/covid_100000.json --max_len 300 --batch_size 4 --num_workers 15 --lr 3e-4 --gpus 1 --max_epochs 4 --save_top_k 2
```
* `python train_script.py --help` to see more info.

### Generation

#### Normal Beam Search

```sh
python generate.py --model_type gpt2 --ckpt lightning_logs/version_4/checkpoints/epoch=3-step=21971.ckpt --prompt 疫苗 --maxlen 500 --num_seq 2
```

* `python generate.py --help` to see more info.

#### Contextual Calibration

* based on this paper: [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2102.09690)


```sh
python generate.py --ckpt gpt2_ckpt/epoch=3  --prompt "日本禁止進口鳳梨"  --maxlen 150 --num_seq 5 --to result_3.txt --model_type gpt2-calibration --device cuda --gpu 0
```

## Bert2Bert
* based on this model: [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
* encode: keyword, decode: body
* 指定 BERT 類型: bert2bert/trainer.py, bert2bert/dataset.py

### finetune
```sh
python train_script.py --model_type bert2bert --train_data data/covid_100000.json --max_len 300 --batch_size 4 --num_workers 15 --lr 3e-4 --gpus 1 --max_epochs 4 --save_top_k 2
```
* `python train_script.py --help` to see more info.

### Generation

```sh
python generate.py --model_type bert2bert --ckpt lightning_logs/version_4/checkpoints/epoch=3-step=21971.ckpt --prompt 疫苗 --maxlen 500 --num_seq 2
```
* `python generate.py --help` to see more info.
