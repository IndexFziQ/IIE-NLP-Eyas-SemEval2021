#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : remove_invalid_question.py
@Time    : 2021/1/26 下午10:43 
@Author  : Luxi Xing
@Contact : xingluxixlx@gmail.com
"""
import os
import json
import linecache
import ast
from tqdm import tqdm
import logging

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger("reformat")

base_dir = "/data1/data-xlx/semeval21_task4/SemEval2021-Task4"
# input_file = os.path.join(base_dir, "training_data/Task_2_train_bktrans_google.jsonl")
# input_file = os.path.join(base_dir, "training_data/Task_2_dev_bktrans_google.jsonl")
input_file = os.path.join(base_dir, "trail_data/Task_2_Nonspecificity_trans_google.jsonl")
# output_file = os.path.join(base_dir, "training_data/Task_2_train_bktrans_google_valid.jsonl")
# output_file = os.path.join(base_dir, "training_data/Task_2_dev_bktrans_google_valid.jsonl")
output_file = os.path.join(base_dir, "trail_data/Task_2_Nonspecificity_trans_google_valid.jsonl")

examples = []
with open(input_file, 'r') as fin:
    for (i, line) in tqdm(enumerate(fin), desc="Loading: "):
        record = json.loads(line.strip())
        bktrans_article = record['bk_article']
        if bktrans_article is None or len(bktrans_article) == 0:
            print(i)
            continue
        original_qst = record['question']
        bktrans_qst = record['bk_question']
        if "@placeholder" in original_qst and "@placeholder" in bktrans_qst:
            examples.append(record)

logger.info("Get original bktrans examples = [{}]".format(len(linecache.getlines(input_file))))
logger.info("Get valid bktrans examples = [{}]".format(len(examples)))
logger.info("Write valid bktrans examples to [{}]".format(output_file))

with open(output_file, 'w') as fout:
    for i, example in tqdm(enumerate(examples), desc="Writing: "):
        fout.write("{}\n".format(json.dumps(example)))

logger.info("Done.")
