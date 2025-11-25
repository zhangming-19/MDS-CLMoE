python3.6 /data/zhangming/BartGraphSumm/src/bart_decode_parallel.py \
--gpus 2 \
--batch_size 16 \
--model /data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model-fewshot-finetuned/checkpoint_best.pt \
--min_len 90 \
--max_len_b 120 \
--task /data/zhangming/data/DUC2004-Ours-1000-remove-docs-bin \
-o /data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model-fewshot-finetuned/test50.decoded /data/zhangming/data/DUC2004-Ours-1000-remove-docs/test50.source

--model /data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model-fewshot-raw/checkpoint_best.pt \
--model /data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model1/checkpoint_best.pt \
