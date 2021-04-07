#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : sim_score_with_bert_finetuned_mrpc.py
@Time    : 2020-12-03 09:53 
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
from transformers import (BertConfig, BertTokenizer, BertForSequenceClassification)


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger("Tool-BERT-sim")

CFG_PATH = "/data1/data-xlx/pytorch-pretrain-lm/pytorch-bert-base-cased-finetuned-mrpc"
MODEL_PATH = "/data1/data-xlx/pytorch-pretrain-lm/pytorch-bert-base-cased-finetuned-mrpc"
TOK_PATH = "/data1/data-xlx/pytorch-pretrain-lm/pytorch-bert-base-cased-finetuned-mrpc"

# DATA_DIR = "/data1/data-xlx/semeval21_task4/SemEval2021-Task4/training_data"
DATA_DIR = "/data1/data-xlx/semeval21_task4/SemEval2021-Task4"

CFG = BertConfig.from_pretrained(CFG_PATH,
                                 finetuning_task="mrpc",
                                 num_labels=2)
TOK = BertTokenizer.from_pretrained(TOK_PATH, do_lower_case=False)
MODEL = BertForSequenceClassification.from_pretrained(MODEL_PATH,
                                                      config=CFG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(device)

def load_examples(input_file_path):
    records = []
    records_only_qs = []
    logger.info(f"Load examples from [{input_file_path}]")
    with open(input_file_path, 'r') as fin:
        for line in fin:
            line = json.loads(line.strip())
            sentences_in_article = line["article"]
            question = line["question"]
            qs_pairs = []
            for _, sent in enumerate(sentences_in_article):
                qs_pairs.append([question, sent])
            records.append(line)
            records_only_qs.append(qs_pairs)
    return records, records_only_qs


def convert_example_to_features(records, tokenizer, max_length=300, pad_token=0, pad_token_segment_id=0):
    logging.info(f"Start to create feature with max length = [{max_length}]")
    features = []
    for (ex_index, example) in tqdm(enumerate(records), desc="creating feature: "):
        feature_input_ids = []
        feature_attention_mask = []
        feature_token_type_ids = []
        for (qs_index, qs) in enumerate(example):
            inputs = tokenizer.encode_plus(qs[0], qs[1],
                                           add_special_tokens=True,
                                           max_length=max_length,
                                           truncation_strategy='only_second',
                                           return_overflowing_tokens=True)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info('* truncated example [{} - {}] tokens number = {}'.format(
                    ex_index,
                    qs_index,
                    inputs['num_truncated_tokens']
                ))
            input_ids, token_type_ids = inputs['input_ids'], inputs["token_type_ids"]
            attention_mask = [1] * len(input_ids)
            # padding
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) ==  max_length
            assert len(attention_mask) ==  max_length
            assert len(token_type_ids) ==  max_length
            feature_input_ids.append(input_ids)
            feature_attention_mask.append(attention_mask)
            feature_token_type_ids.append(token_type_ids)
        features.append([feature_input_ids, feature_attention_mask, feature_token_type_ids])
    return features


def rank_logits(example, prediction_prob):
    ranked_sentence = []
    num_of_sentence = len(example)
    assert len(prediction_prob) == num_of_sentence
    ranked_prob = sorted(enumerate(prediction_prob), key=lambda x: x[1], reverse=True)
    for rank_id, (s_id, s_prob) in enumerate(ranked_prob):
        each_sentence = {}
        each_sentence['rank_index'] = rank_id
        each_sentence['original_index'] = s_id
        each_sentence['similarity_prob'] = s_prob
        each_sentence['sentence'] = example[s_id][1]
        ranked_sentence.append(each_sentence)

    return ranked_sentence


def compute_similarity_label(model, features, records_qs):
    ranked_records_article = []
    for (ex_index, feature) in tqdm(enumerate(features), desc="Computing: "):
        input_ids = torch.tensor(feature[0], dtype=torch.long).to(device)
        attention_mask = torch.tensor(feature[1], dtype=torch.long).to(device)
        token_type_ids = torch.tensor(feature[2], dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs[0].detach().cpu().numpy()
            predict_prob = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)
            # print(predict_prob)
            # get score for similarity
            predict_prob_for_paraphrase = torch.split(predict_prob, split_size_or_sections=1, dim=-1)[1].tolist()
            ranked_sentence = rank_logits(records_qs[ex_index], predict_prob_for_paraphrase)
            ranked_records_article.append(ranked_sentence)
    return ranked_records_article


def update_output_file(output_file_path, ranked_records_article, records):
    logger.info(f"Write update examples to [{output_file_path}]")
    with open(output_file_path, 'w') as fout:
        for ri, record in tqdm(enumerate(records), desc="Write updated examples: "):
            new_record = record
            new_record['article_similarity_score'] = [s['similarity_prob'] for s in ranked_records_article[ri]]
            new_record['article_original_sentence_index'] = [s['original_index'] for s in ranked_records_article[ri]]
            new_record['article_ranked'] = [s['sentence'] for s in ranked_records_article[ri]]
            fout.write("{}\n".format(json.dumps(new_record)))
    return


if __name__ == '__main__':
    # input_files = ["Task_1_train_split.jsonl", "Task_1_dev_split.jsonl"]
    # input_files = ["Task_2_train_split.jsonl", "Task_2_dev_split.jsonl"]
    input_files = ["trail_data/Task_1_Imperceptibility_split.jsonl",
                   "trail_data/Task_2_Nonspecificity_split.jsonl",
                   "trail_data/Task_3_Intersection_split.jsonl"]

    for input_file_name in input_files:
        print("##########################################")
        output_file_name = input_file_name.replace("split", "bert_mrpc_rank")

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
        features = convert_example_to_features(records_only_qs, TOK)
        logging.info(f"Load features numer = [{len(features)}]")

        ranked_records_article = compute_similarity_label(MODEL, features, records_only_qs)

        update_output_file(output_file, ranked_records_article, records)

    logger.info("Done")
