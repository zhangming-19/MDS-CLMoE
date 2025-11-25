#!/usr/bin/env python
##  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
##  SPDX-License-Identifier: MIT

import argparse
import os
from nltk.tokenize import sent_tokenize
from graph_utils import cluster_docs
from multiprocessing import Pool
from tqdm import tqdm
import json
import numpy as np
import random
import math
from rouge import Rouge
#import ipdb

random.seed(111)

# SPLITs = ['train', 'val', 'test']
SPLITs = ['test']
GRAPH_SENT_LIMIT = 100
RANKING_OFFSET = 2

SEPARATER_TAG = "</s>"


def read_tgt(filename):
    """ module to read the target data and 
    send it as list of tgts"""
    print(f"Reading the target file: {filename}")
    tgts = []
    with open(filename, "r") as f:
        for line in f.read().splitlines():
            tgts.append(line.strip())
    
    print("Printing a sample from this file")
    print(f"{tgts[0][:50]}[...]")

    return tgts

 
def read_src(filename):
    """ module to read the target data and 
    send it as list of tgts"""
    print(f"Reading the source file: {filename}")
    srcs = []
    with open(filename, "r") as f:
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


def create_data(args):
    if not os.path.exists(args.output_path):
        print(f"Creating directory: {args.output_path}")
        os.mkdir(args.output_path)

    for split in SPLITs:
        print(f"Processing Split: {split}")
        srcs = read_src(os.path.join(args.data_path, f"{split}.source"))
        tgts = read_tgt(os.path.join(args.data_path, f"{split}.target"))

        if args.graph_encoding:
            graph_info = read_graph(os.path.join(args.graph_data_path, f"{split}.jsonl"))

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

        # Truncate the source
        new_srcs = []
        for src in srcs:
            
            src = " story_separator_special_tag ".join(src) 
            src = truncate(src, total_words=args.max_length)
            src = src.split(" story_separator_special_tag ")
            new_srcs.append(src)

        f_src = os.path.join(args.output_path, f"{split}.source")
        f_tgt = os.path.join(args.output_path, f"{split}.target")

        if args.sentence_level_markers:
            if args.graph_encoding:
                new_srcs_g = []
                for index, src in enumerate(new_srcs):
                    scores_list = [v["score"] for k,v in graph_info[index].items()]
                    threshold1 = np.quantile(np.array([scores_list]), 0.33)
                    threshold2 = np.quantile(np.array([scores_list]), 0.67)
                    new_docs = []
                    for i, doc in enumerate(src):
                        new_doc = []
                        for j, sent in enumerate(sent_tokenize(doc)):
                            if j < GRAPH_SENT_LIMIT:
                                id_ = "d{}_s{}".format(i,j)
                                score = graph_info[index][id_]["score"]
                                if score>threshold2:
                                    label = "high"
                                elif score>threshold1:
                                    label = "medium"
                                else:
                                    label = "low"
                                new_doc.append(sent + f" graph score is {label} {SEPARATER_TAG}")
                            else:
                                new_doc.append(sent + f" {SEPARATER_TAG}")
                        new_docs.append(" ".join(new_doc))
                    new_srcs_g.append(" ".join(new_docs))
                srcs = new_srcs_g
            else:      
                #TODO: right now the joining of docs is bad as src tokenize gets wrong
                srcs = [" ".join(src) for src in new_srcs] 
                srcs = [sent_tokenize(src) for src in srcs]
                srcs = [f" {SEPARATER_TAG} ".join(src) for src in srcs]

        else:
            srcs = [f" {SEPARATER_TAG} ".join(src) for src in new_srcs]


        with open(f_src, "w") as f:
            f.write("\n".join(srcs))
            f.flush()
            f.close()

        with open(f_tgt, "w") as f:
            f.write("\n".join(tgts))
            f.flush()
            f.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/data/multi-news-full-clean')
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--graph_data_path', type=str, default='/home/ubuntu/data/multi-news-full-clean-graph-pagerank-tfidf')
    parser.add_argument('--output_path', type=str, default='/home/ubuntu/data/multi-news-500')
    parser.add_argument('--max_length', type=int, default=500)
    parser.add_argument('--sentence_level_markers', action='store_true', default=False)
    parser.add_argument('--graph_encoding', type=str, default='', help="if not empty, \
                        added some graph info to the inputs in textual form")
    parser.add_argument('--similarity_metric', type=str, default='tfidf', help="choose option like rouge-1/2/l or tfidf")
    parser.add_argument('--num_workers', type=int, default=1, help="set this number of multiprocessing threads")
    parser.add_argument('--num_batches', type=int, default=1, help="set this number for processing/saving outputs in chunks")
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="standard")
    parser.add_argument('--shuffle_sentences', action='store_true', default=False)
    parser.add_argument('--start_chunk_index', type=int, default=0)
    parser.add_argument('--special_linearize', action='store_true', default=False)

    args = parser.parse_args()
    print(args)


    # create_data(args)
    srcs = read_src(os.path.join(args.data_path, f"test.source"))
    print(len(srcs[0]))

    
