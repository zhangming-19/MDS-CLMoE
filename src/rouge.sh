python3.6 /data/zhangming/PreSumm/src/cal_rouge_old.py \
-p 32 \
-r /data/zhangming/data/DUC2004-Ours-1000-remove-docs/test50.target \
-c /data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model-fewshot-raw/test.decoded


-c /data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model1/DUC2004.decoded 
#-c /data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model-finetuned-raw/test.decoded 

#2>/data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model1/DUC2004.rouge-stderr \
#>/data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model1/DUC2004.rouge-stdout

# -r /data/zhangming/data/DUC2004-Ours-1000-remove-docs/DUC2004.target \
#-c /data/zhangming/results/bart-large-duc2004-1000-wo-sentmarkers-model1/DUC2004.decoded \