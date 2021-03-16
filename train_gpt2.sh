python train_script.py --model_type gpt2 --train_data data/all.json --max_len 250 --batch_size 16 --num_workers 15 --lr 3e-4 --gpus 1 --max_epochs 3 --save_top_k 4 \
--keywords_loss_fct kldiv \
--keywords_loss_alpha 0.9 \
--with_keywords_loss 