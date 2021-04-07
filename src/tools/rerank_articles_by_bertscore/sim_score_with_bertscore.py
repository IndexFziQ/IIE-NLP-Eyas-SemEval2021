#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : retrieve_with_bertscore.py
@Time    : 2020-12-03 09:52 
@Author  : Luxi Xing
@Contact : xingluxixlx@gmail.com
"""
import os
import sys
import json
import logging
from tqdm import tqdm
import torch

sys.path.append("..")
sys.path.append("../")
print(sys.path)
from transformers import (BertConfig, BertTokenizer, BertModel)
from bert_score import score
from sim_score_with_bert_finetuned_mrpc import (load_examples,update_output_file)

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger("Tool-BERT-sim")

MODEL_PATH = "/data1/data-xlx/pytorch-pretrain-lm/bert-large-uncased"
# DATA_DIR = "/data1/data-xlx/semeval21_task4/SemEval2021-Task4/training_data"
DATA_DIR = "/data1/data-xlx/semeval21_task4/SemEval2021-Task4"

CFG = BertConfig.from_pretrained(MODEL_PATH)
MODEL = BertModel.from_pretrained(MODEL_PATH, config=CFG)
TOK = BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)
TOK.max_len=512

def rank_bertscore(example, cal_bertscore):
    ranked_sentence = []
    num_of_sentence = len(example)
    assert len(cal_bertscore) == num_of_sentence
    ranked_prob = sorted(enumerate(cal_bertscore), key=lambda x: x[1], reverse=True)
    for rank_id, (s_id, s_prob) in enumerate(ranked_prob):
        each_sentence = {}
        each_sentence['rank_index'] = rank_id
        each_sentence['original_index'] = s_id
        each_sentence['similarity_prob'] = s_prob
        each_sentence['sentence'] = example[s_id][1]
        ranked_sentence.append(each_sentence)

    return ranked_sentence


def compute_bertscore(records_qs):
    ranked_records_article = []
    for (ex_index, qs) in tqdm(enumerate(records_qs), desc="Computing: "):
        q = [qsi[0] for qsi in qs]
        s = [qsi[1] for qsi in qs]
        (p, r, f), hashname = score(s, q,
                                    model_type=MODEL_PATH,
                                    lang='en',
                                    num_layers=18,
                                    return_hash=True,
                                    loaded_tokenizer=TOK,
                                    loaded_model=MODEL)
        cal_bertscore = f.tolist()
        ranked_sentence = rank_bertscore(records_qs[ex_index], cal_bertscore)
        ranked_records_article.append(ranked_sentence)
    return ranked_records_article


if __name__ == '__main__':
    # input_files = ["Task_1_train_split.jsonl", "Task_1_dev_split.jsonl"]
    # input_files = ["Task_2_train_split.jsonl", "Task_2_dev_split.jsonl"]
    # input_files = ["trail_data/Task_1_Imperceptibility_split.jsonl",
    #                "trail_data/Task_2_Nonspecificity_split.jsonl",
    #                "trail_data/Task_3_Intersection_split.jsonl"]
    input_files = ["test_data/Task_2_test_split.jsonl"]

    for input_file_name in input_files:
        print("##########################################")
        output_file_name = input_file_name.replace("split", "bertscore_rank")

        input_file = os.path.join(
            DATA_DIR,
            input_file_name
        )
        output_file = os.path.join(
            DATA_DIR,
            output_file_name
        )

        records, records_only_qs = load_examples(input_file)
        logger.info(f"Load examples number = [{len(records)}]")

        ranked_records_article = compute_bertscore(records_only_qs)

        update_output_file(output_file, ranked_records_article, records)

    logger.info("Done")
