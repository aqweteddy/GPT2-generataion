python train_script.py --model_type bert2bert \
--keywords_loss_fct kldiv \
--keywords_loss_alpha 0.9 \
--with_keywords_loss \
--exp_name "all.json (keyword loss) kldiv" \
--train_data data/all.json \
--max_len 150 \
--title_max_len 15 \
--batch_size 16 \
--num_workers 10 \
--lr 2e-5 \
--gpus 1 \
--max_epochs 5 \
--save_top_k 6 \
--gpuid 0

# --with_keywords_loss \