# init rag ckpt
# python rag/consolidate_rag_checkpoint.py \
#             --model_type rag_sequence \
#             --generator_name_or_path rag_ckpt/rag_generator/albert2albert \
#             --question_encoder_name_or_path ckiplab/albert-base-chinese \
#             --generator_tokenizer_name_or_path bert-base-chinese \
#             --question_encoder_tokenizer_name_or_path bert-base-chinese \
#             --dest rag_ckpt/rag_init

python rag/create_db.py \
            --csv_path data/wiki/wiki.csv \
            --rag_model_name rag_ckpt/rag_init \
            --ctx_encoder_model_name ckiplab/albert-base-chinese \
            --ctx_encoder_tokenizer_name bert-base-chinese \
            --output_dir rag_ckpt/wiki_tw_index


# create db
