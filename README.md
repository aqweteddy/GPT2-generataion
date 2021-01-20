# README

* based on `pytorch` / `transformers` and `pytorch_lightning`

## GPT2

* pretrained model: CKIP Lab GPT2-base

### fine-tune

```sh
python train_script.py --train_data data/covid_100000.json --max_len 1000 --batch_size 4 --num_workers 15 --lr 3e-4 --gpus 1 --max_epochs 4 --save_top_k 2
```
* `python train_script.py --help` to see more info.

### Generation

```sh
python generate.py --ckpt lightning_logs/version_4/checkpoints/epoch=3-step=21971.ckpt --prompt 疫苗 --maxlen 500 --num_seq 2
```
* `python generate.py --help` to see more info.

## Bert2Bert

* encode: title, decode: body