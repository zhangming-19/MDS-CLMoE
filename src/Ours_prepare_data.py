#!/usr/bin/env python
##  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
##  SPDX-License-Identifier: MIT

import argparse
import sys
import os
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize
# from graph_utils import cluster_docs
from multiprocessing import Pool
from tqdm import tqdm
import json
import codecs
import numpy as np
import pandas as pd
import random
import math
import rouge 
import torch
import copy
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
#import ipdb
# import bert_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(111)

# SPLITs = ['train', 'val', 'test']
# SPLITs = ['test']
SPLITs = ['test-small']

GRAPH_SENT_LIMIT = 100
RANKING_OFFSET = 2
SEPARATER_TAG = "</s>"
VALUE_ALPHA = 0.5
VALUE_BETA = 0.5
VALUE_GAMMA = 0.5

# Load SimCSE Model
SimCSE_tokenizer = AutoTokenizer.from_pretrained("./princeton-nlp.sup-simcse-roberta-large")
SimCSE_model = AutoModel.from_pretrained("./princeton-nlp.sup-simcse-roberta-large")

# Load Rouge evaluator
apply_avg = True
apply_best = False
evaluator = rouge.Rouge(metrics=['rouge-n'],
                        max_n=4,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=apply_avg,
                        apply_best=apply_best,
                        alpha=0.5,  # Default F1_score
                        weight_factor=1.2,
                        stemming=True)


####################### FUNCTION ######################
def read_tgt(filename):
    """ module to read the target data and 
    send it as list of tgts"""
    print(f"Reading the target file: {filename}")
    tgts = []
    with open(filename, "r", encoding='utf-8') as f:
        for line in f.read().splitlines():
            tgts.append(line.strip())
    
    print("Printing a sample from this file")
    print(f"{tgts[0][:50]}[...]")

    return tgts


####################### FUNCTION ###################### 
def read_src(filename):
    """ module to read the target data and 
    send it as list of tgts"""
    print(f"Reading the source file: {filename}")
    srcs = []
    with open(filename, "r", encoding='utf-8') as f:
        for line in f.read().splitlines():
            line = " ".join(line.split())
            docs = line.split("story_separator_special_tag")  
            docs = [d.strip() for d in docs]
            srcs.append(docs)

    print("Printing a sample from this file")
    print(f"Total Documents in this sample: {len(srcs[0])}")
    print("#"*20)
    for d in srcs[0]:
        print(f"{d[:50]}[...]")
    print("#"*20)

    return srcs


####################### FUNCTION ######################
def truncate(line, separator_tag="story_separator_special_tag", total_words=500):
    line_word_split = line.split()
    if len(line_word_split) < total_words:
        return line
    else:
        sources_split = line.split(separator_tag)
        # previous dataset had separator at the end of each example
        if sources_split[-1] == "":
            del sources_split[-1]

        num_sources = len(sources_split)
        words_ar = [source.split() for source in sources_split]
        num_words_ar = [len(words) for words in words_ar]
        #logging.debug(f"initial number of words: {str(num_words_ar)}")
        per_source_count = math.floor(total_words / num_sources)
        total_ar = [0] * num_sources
        total = 0
        done = {}
        while total < total_words and len(done) < len(num_words_ar):
            # e.g. total=499 and still trying to add -- just add from the first doc which isn't done
            if per_source_count == 0:
                for index, x in enumerate(total_ar):
                    if index not in done:
                        total_ar[index] += total_words - total
                        break
                break
            min_amount = min(min([x for x in num_words_ar if x > 0]), per_source_count)
            total_ar = [x + min_amount if index not in done else x for index, x in enumerate(total_ar)]
            for index, val in enumerate(num_words_ar):
                if val == min_amount:
                    done[index] = True
            num_words_ar = [x - min_amount for x in num_words_ar]
            total = sum(total_ar)
            if len(done) == len(num_words_ar):
                break
            per_source_count = math.floor((total_words - total) / (len(num_words_ar) - len(done))) 
        final_words_ar = []
        for count_words, words in enumerate(words_ar):
            cur_string = " ".join(words[:total_ar[count_words]])
            final_words_ar.append(cur_string)
        final_str = (" " + separator_tag + " ").join(final_words_ar).strip()
        
        return final_str


####################### FUNCTION ######################
def calculate_similarity(text_list:list):
    # Tokenize input texts    
    inputs = SimCSE_tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    Total = len(text_list)

    # Get the SimCSE_embeddings
    with torch.no_grad():
        SimCSE_embeddings = SimCSE_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    results_list = []
    for i in range(1,Total):
        # print(i)
        cosine_sim = 1 - cosine(SimCSE_embeddings[0], SimCSE_embeddings[i])
        results_list.append(cosine_sim)
    
    return results_list

    
####################### FUNCTION ######################
def calculate_rouge(text_list:list):
    Total = len(text_list)
    references = text_list[0].split('\n')
    references = [line.strip() for line in references]
    
    results_list = []
    for i in range(1,Total):
        candidates = text_list[i].split('\n')
        candidates = [line.strip() for line in candidates]
        # print(candidates)
        # print(len(candidates))
        # print('*'*20)
        # print(references)
        # print(len(references))
        assert len(candidates) == len(references)
        
        all_hypothesis = candidates
        all_references = references
        scores = evaluator.get_scores(all_hypothesis, all_references)

        rougel = 0
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if metric in ["rouge-1"]:
        #        print(prepare_results(metric, results['p'], results['r'], results['f']))
                rougel = float(results['f'])
        results_list.append(rougel)

    return results_list


####################### FUNCTION ######################
def remove_redundant_sentences(text_list:list, similarity_list:list, rouge1_list:list):
    Total = len(text_list)
    base_text = text_list[0]
    base_sent_list = sent_tokenize(base_text)
    M = len(base_sent_list)
    
    important_list = []
    new_text_list = []
    new_text_list.append(base_text)
    
    # for k in range(1,Total):
    num = 0
    for item_similarity, item_rouge1 in zip(similarity_list, rouge1_list):
        num = num + 1
        redundant_text = text_list[num]
        redundant_sent_list = sent_tokenize(redundant_text)  
        temp_redundant_sent_list = copy.deepcopy(redundant_sent_list)  
        N = len(redundant_sent_list)
        
        if item_similarity >= VALUE_ALPHA and item_rouge1 >= VALUE_BETA:
            redundant_list = []
            for m in range(M):
                for n in range(N):
                    references = base_sent_list[m].split('\n')
                    references = [line.strip() for line in references]
                    candidates = redundant_sent_list[n].split('\n')
                    candidates = [line.strip() for line in candidates]
                    assert len(candidates) == len(references)
                    
                    all_hypothesis = candidates
                    all_references = references
                    scores = evaluator.get_scores(all_hypothesis, all_references)
                    rougel = 0
                    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                        if metric in ["rouge-1"]:
                            rougel = float(results['f'])
                    
                    # output_matrix[i,j] = rougel
                    # 去除冗余句子，标注重点句子
                    if rougel >= VALUE_GAMMA:
                        important_list.append(m)
                        redundant_list.append(n)
            
            # print('#'*20)
            # print(redundant_list)
            redundant_list = list(set(redundant_list))
            redundant_list.sort(reverse=False)
            for num, i in enumerate(redundant_list):
                # print(i-num)
                temp_redundant_sent_list.pop(i-num)
                
            temp_text = " ".join(temp_redundant_sent_list)
            new_text_list.append(temp_text)
        
        else:
            important_list.append(0)
            new_text_list.append(redundant_text)
            
        # important_list = list(set(important_list))
        # important_list = important_list.sort(reverse=False)
        
        list(set(important_list)).sort(reverse=False)

    return important_list, new_text_list


####################### FUNCTION ######################
def sent_remove_sent(str_1:str, str_2:str):
    list_str_1 = sent_tokenize(str_1)
    list_str_2 = sent_tokenize(str_2)
    
    for item in list_str_2:
        list_str_1.remove(item)
    
    new_str_1 = ' '.join(list_str_1)
    return new_str_1




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./multi-news-full-clean')
    parser.add_argument('--output_path', type=str, default='./multi-news-Ours')
    parser.add_argument('--max_length', type=int, default=500)
    parser.add_argument('--sentence_level_markers', action='store_true', default=False)
    parser.add_argument('--shuffle_sentences', action='store_true', default=False)

    args = parser.parse_args()
    args.sentence_level_markers = True
    # if args.mode == "create_graphs":
    #     create_graphs(args)
    # elif args.mode == "standard_with_graph_knowledge":
    #     create_data_with_graph_knowledge(args)
    # else:
    #     create_data(args)
    
    
    # for split in SPLITs:
    #     print(f"Processing Split: {split}")
    #     raw_srcs = read_src(os.path.join(args.data_path, f"{split}.source"))
        
    #     # conut raw average_input
    #     num_input = []
    #     for src in raw_srcs:
    #         COUNT = 0
    #         for item in src:
    #             COUNT += len(item.split(' '))
    #         num_input.append(COUNT)
            
    #     average_input = int(sum(num_input)/len(num_input))
    #     print(f"###############", split, f"###############")
    #     print(f"Original average_input:::", average_input)
        
        
    if not os.path.exists(args.output_path):
        print(f"Creating directory: {args.output_path}")
        os.mkdir(args.output_path)


    for split in SPLITs:
        print(f"Processing Split: {split}")
        srcs = read_src(os.path.join(args.data_path, f"{split}.source"))
        tgts = read_tgt(os.path.join(args.data_path, f"{split}.target"))
        
        ################### 篇章句子打乱顺序 #######################
        if args.shuffle_sentences:
            print("shuffling the sentences within a document")
            new_srcs = []
            for ind, src in enumerate(srcs):
                new_docs = []
                for doc in src:
                    doc_sents = sent_tokenize(doc)
                    random.shuffle(doc_sents)
                    new_docs.append(" ".join(doc_sents))
                new_srcs.append(new_docs)
                if ind==0:
                    print(f"Sample original src:::", srcs[0])
                    print(f"Sample sentence shuffled src:::", new_srcs[0])

            srcs = new_srcs
        
        #################### 重新排序，最长的字符串作为base，排在首位 ###################
        for src in srcs:
            base_num = src.index(max(src, key=len, default=''))
            if base_num == 0:
                pass
            else:
                temp = src[base_num]
                src.pop(base_num)
                src.insert(0, temp)

        ################### 处理多文档冗余 ###################
        output_similarity = []
        output_rouge1 = []
        output_important = []
        output_srcs = []
        for src in srcs:
            text_first = []
            text_all = []
            similarity_list = []
            rouge1_list = []
            important_temp = []
            output_temp = []
            text_first.append(sent_tokenize(src[0])[0])
            text_all.append(src[0])
            
            if len(src) == 1:   
                similarity_list.append('None')
                rouge1_list.append('None')

            elif len(src) == 2:
                text_first.append(sent_tokenize(src[1])[0])
                similarity_list = calculate_similarity(text_first)
                text_all.append(src[1])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)

            elif len(src) == 3:
                text_first.append(sent_tokenize(src[1])[0])
                text_first.append(sent_tokenize(src[2])[0])
                similarity_list = calculate_similarity(text_first)
                text_all.append(src[1])
                text_all.append(src[2])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)
                
            elif len(src) == 4:
                text_first.append(sent_tokenize(src[1])[0])
                text_first.append(sent_tokenize(src[2])[0])
                text_first.append(sent_tokenize(src[3])[0])
                similarity_list = calculate_similarity(text_first)
                text_all.append(src[1])
                text_all.append(src[2])
                text_all.append(src[3])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)
            
            elif len(src) == 5:
                text_first.append(sent_tokenize(src[1])[0])
                text_first.append(sent_tokenize(src[2])[0])
                text_first.append(sent_tokenize(src[3])[0])
                text_first.append(sent_tokenize(src[4])[0])
                similarity_list = calculate_similarity(text_first)
                text_all.append(src[1])
                text_all.append(src[2])
                text_all.append(src[3])
                text_all.append(src[4])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)
            
            elif len(src) == 6:            
                text_first.append(sent_tokenize(src[1])[0])
                text_first.append(sent_tokenize(src[2])[0])
                text_first.append(sent_tokenize(src[3])[0])
                text_first.append(sent_tokenize(src[4])[0])
                text_first.append(sent_tokenize(src[5])[0])
                similarity_list = calculate_similarity(text_first)
                text_all.append(src[1])
                text_all.append(src[2])
                text_all.append(src[3])
                text_all.append(src[4])
                text_all.append(src[5])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)
                
            elif len(src) == 7:            
                text_first.append(sent_tokenize(src[1])[0])
                text_first.append(sent_tokenize(src[2])[0])
                text_first.append(sent_tokenize(src[3])[0])
                text_first.append(sent_tokenize(src[4])[0])
                text_first.append(sent_tokenize(src[5])[0])
                text_first.append(sent_tokenize(src[6])[0])
                similarity_list = calculate_similarity(text_first)
                text_all.append(src[1])
                text_all.append(src[2])
                text_all.append(src[3])
                text_all.append(src[4])
                text_all.append(src[5])
                text_all.append(src[6])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)
                
            elif len(src) == 8:            
                text_first.append(sent_tokenize(src[1])[0])
                text_first.append(sent_tokenize(src[2])[0])
                text_first.append(sent_tokenize(src[3])[0])
                text_first.append(sent_tokenize(src[4])[0])
                text_first.append(sent_tokenize(src[5])[0])
                text_first.append(sent_tokenize(src[6])[0])
                text_first.append(sent_tokenize(src[7])[0])
                similarity_list = calculate_similarity(text_first)
                text_all.append(src[1])
                text_all.append(src[2])
                text_all.append(src[3])
                text_all.append(src[4])
                text_all.append(src[5])
                text_all.append(src[6])
                text_all.append(src[7])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)
            
            elif len(src) == 9:           
                text_first.append(sent_tokenize(src[1])[0])
                text_first.append(sent_tokenize(src[2])[0])
                text_first.append(sent_tokenize(src[3])[0])
                text_first.append(sent_tokenize(src[4])[0])
                text_first.append(sent_tokenize(src[5])[0])
                text_first.append(sent_tokenize(src[6])[0])
                text_first.append(sent_tokenize(src[7])[0])
                text_first.append(sent_tokenize(src[8])[0])
                similarity_list = calculate_similarity(text_first)
                text_all.append(src[1])
                text_all.append(src[2])
                text_all.append(src[3])
                text_all.append(src[4])
                text_all.append(src[5])
                text_all.append(src[6])
                text_all.append(src[7])
                text_all.append(src[8])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)
            
            elif len(src) == 10:
                text_first.append(sent_tokenize(src[1])[0])
                text_first.append(sent_tokenize(src[2])[0])
                text_first.append(sent_tokenize(src[3])[0])
                text_first.append(sent_tokenize(src[4])[0])
                text_first.append(sent_tokenize(src[5])[0])
                text_first.append(sent_tokenize(src[6])[0])
                text_first.append(sent_tokenize(src[7])[0])
                text_first.append(sent_tokenize(src[8])[0])
                text_first.append(sent_tokenize(src[9])[0])
                similarity_list = calculate_similarity(text_first)  
                text_all.append(src[1])
                text_all.append(src[2])
                text_all.append(src[3])
                text_all.append(src[4])
                text_all.append(src[5])
                text_all.append(src[6])
                text_all.append(src[7])
                text_all.append(src[8])
                text_all.append(src[9])
                rouge1_list = calculate_rouge(text_all)
                assert len(similarity_list) == len(rouge1_list)
                important_temp, output_temp = remove_redundant_sentences(text_all, similarity_list, rouge1_list)

            output_similarity.append(similarity_list)
            output_rouge1.append(rouge1_list)
            output_important.append(important_temp)
            output_srcs.append(output_temp)
            
                
        name = ['similarity', 'rouge1']
        temp = []
        temp.append(output_similarity)
        temp.append(output_rouge1)
        temp_df = np.array(temp)
        temp_df = temp_df.T
        temp_df = pd.DataFrame(temp_df, columns=name)
        temp_df.to_csv('analysis_data.csv', encoding='utf-8')
                
        test=pd.DataFrame(data=output_important)
        test.to_csv('output_important.csv',encoding='utf-8')

        test=pd.DataFrame(data=output_srcs)
        test.to_csv('output_srcs.csv',encoding='utf-8')
        
        # df = pd.read_csv('./raw_data.csv')
        # df2 = pd.read_csv('./output_important.csv')
        # df3 = pd.read_csv('./output_srcs.csv')
   
    
        ################### 开始截取数据 ###################
        # print(output_important)
        # Truncate the source
        new_srcs = []
        for important, src in zip(output_important, output_srcs):
            important = list(set(important))
            important.sort(reverse=False)
            # print(important)
            
            temp_text = src[0]
            temp_list = sent_tokenize(temp_text)
            important_list = []
            for item in important:
                important_list.append(temp_list[item])
            
            important_text = " ".join(important_list)
            num_important_words = len(important_text.split(' '))
            # print(important_text)
            
            if num_important_words < args.max_length:
                total_words = args.max_length - num_important_words   
                src = " story_separator_special_tag ".join(src) 
                
                src = sent_remove_sent(src, important_text)
                src = truncate(src, total_words=total_words)
                
                
                src_list = list(src)
                src_list.insert(0, important_text)
                src = ''.join(src_list)
                
                src = src.split(" story_separator_special_tag ")            
                src = [item + '.' for item in src]                
                new_srcs.append(src)
            else:
                src = important_text.split(' ')[:args.max_length]
                src = ' '.join(src)
                new_srcs.append([src])



        srcs = [" ".join(src) for src in new_srcs] 
        # srcs = [sent_tokenize(src) for src in srcs]
        # srcs = [f" {SEPARATER_TAG} ".join(src) for src in srcs]


        f_src = os.path.join(args.output_path, f"{split}.source")
        f_tgt = os.path.join(args.output_path, f"{split}.target")

        with open(f_src, "w", encoding='utf-8') as f:
            f.write("\n".join(srcs))
            f.flush()
            f.close()

        with open(f_tgt, "w", encoding='utf-8') as f:
            f.write("\n".join(tgts))
            f.flush()
            f.close()         
    



