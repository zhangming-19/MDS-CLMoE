# Towards curriculum learning of multi-document summarization using difficulty-aware mixture-of-experts

This project includes the source code for the paper [**Towards curriculum learning of multi-document summarization using difficulty-aware mixture-of-experts**](https://www.sciencedirect.com/science/article/pii/S1568494625014012), appearing at Applied Soft Computing. Please cite this [article](https://doi.org/10.1016/j.asoc.2025.114088) as follows, if you use this code.

> Zhang M, Cheng L, Guan W, et al. Towards curriculum learning of multi-document summarization using difficulty-aware mixture-of-experts[J]. Applied Soft Computing, 2025: 114088.

**Highlighted Features**
* Our work demonstrates the efficacy of curriculum learning in multi-document summarization.
* Our work highlights the strength of mixture-of-experts in multi-document summarization.
* CLMoE: a novel framework that incorporates Curriculum Learning into Mixture-of-Experts.
* Evaluation on two MDS benchmark datasets demonstrates competitive performance of CLMoE.

## Requirements
We use Conda python 3.6 and strongly recommend that you create a new environment.
* Prerequisite: Python 3.6 or higher versions
```shell script
conda create -n MDS-CLMoE python=3.6
conda activate MDS-CLMoE
```

## Environment
This code is tested using Python 3.6, Pytorch 1.10, and CUDA 11.1
* Install all packages in the requirement.txt
```shell script
pip3 install -r requirements.txt
```

Setup Up for fairSeq:
```shell script
git clone --branch v0.9.0 https://github.com/pytorch/fairseq
cd ~/fairseq
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```

Set Up for ROUGE: Read more from this [link](https://github.com/bheinzerling/pyrouge).
```shell script
pip install -U git+https://github.com/pltrdy/pyrouge
git clone https://github.com/pltrdy/files2rouge.git
cd files2rouge
python setup_rouge.py
python setup.py install
pyrouge_set_rouge_path ~/.files2rouge
```

## Datasets
### Multi-News
More details can be find in this [link](https://github.com/Alex-Fabbri/Multi-News). please request and download the data from the original paper.
* [multi-news-500](https://drive.google.com/drive/folders/1qqSnxiaNVEctgiz2g-Wd3a9kwWuwMA07) (Preprocessed and truncated data): rename the folder as multi-news-500; 
* [multi-news-full-clean](https://drive.google.com/drive/folders/1qZ3zJBv0zrUy4HVWxnx33IsrHGimXLPy) (Preprocessed but not truncated): rename the folder as multi-news-full-clean; 
* [multi-news-ours](https://drive.google.com/drive/folders/1qgV_tZA3lbr9J5bhOLwmi4YIvU0Nqrx-?usp=sharing): rename the folder as multi-news-ours; 

### DUC-2004
More details can be find in this [link](https://github.com/UsmanNiazi/DUC-2004-Dataset). please request and download the data from the original paper.

## Train and Evaluate
### Command-line interface
```shell script
cd ~/src
make -f bart-large-ours.mk TASK=~/data/multi-news-ours OUTPUT_DIR=~/results/bart-large-multinews-model-ours rouge
```
The ROUGE F1 scores (R1/R2/RL) can be found at test.rouge-stdout

## Citation
```
@article{ZHANG2026114088,
title = {Towards curriculum learning of multi-document summarization using difficulty-aware mixture-of-experts},
journal = {Applied Soft Computing},
volume = {186},
pages = {114088},
year = {2026},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2025.114088},
url = {https://www.sciencedirect.com/science/article/pii/S1568494625014012},
author = {Ming Zhang and Lu Cheng and Wenbo Guan and Jun Zhou and Meilin Wan},
keywords = {Multi-document summarization, Text summarization, Curriculum learning, Mixture-of-experts, Natural language processing},
abstract = {With the advent of information explosion, automatic multi-document summarization has attracted widespread attention from the natural language processing. Multi-document summarization (MDS) poses challenges with larger search and problem spaces, since lengthy inputs often contain varying degrees of redundancy and contradiction. To tackle these challenges, we propose a novel difficulty-aware framework that enhances MDS by integrating curriculum learning paradigms into mixture-of-experts architectures. During the curriculum learning stage, a multi-document difficulty metric works in tandem with the curriculum scheduler to train experts specialized in distinct problem subspaces. We first employ the proposed multi-document difficulty metric to partition the MDS dataset into subsets with distinct levels of difficulty. Then, we adopt the difficulty-aware curriculum arrangement to train specialized expert models on different subsets. During the mixture-of-experts stage, we utilize a difficulty-aware mixture-of-experts structure to combine different models for improving multi-document summarization. Extensive experimental results on two MDS datasets indicate that the proposed method achieves state-of-the-art performance among length-constrained methods and delivers competitive results compared to other strong baselines with greater parameters. These results demonstrate the effectiveness of combining curriculum learning and mixture-of-experts, implying a promising direction for multi-document summarization.}
}
```

## Get Involved
Should you have any query please contact me at [zhangming@hccl.ioa.ac.cn](mailto:zhangming@hccl.ioa.ac.cn).
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. 
Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.
