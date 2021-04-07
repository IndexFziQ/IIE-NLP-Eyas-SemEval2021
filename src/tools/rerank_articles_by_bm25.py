#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : re_rank_articles_bm25.py
@Author : Yuqiang Xie
@Date   : 2020/12/2
@E-Mail : indexfziq@gmail.com
"""
import nltk
from nltk.corpus import stopwords
from gensim.summarization import bm25
import jsonlines
import os

def cut_stopwords(word_list):

    stopwords_ = stopwords.words('english')
    for w in ['!',',','.','``',"''",'?','-s','-ly','</s>','s']:
        stopwords_.append(w)

    filtered_words = [word for word in word_list if word not in stopwords_]

    return filtered_words

def tokenization(sent):

    word_list = nltk.word_tokenize (sent)
    word_list = cut_stopwords(word_list)

    result = []
    for word in word_list:
        result.append(word)

    return result

def FindList2MaxNum(list):

    a , b = (list[0],list[1]) if list[0] > list[1] else (list[1],list[0])
    for i in range(2,len(list)):
        if list[i] > list[0]:
            b = a
            a = list[i]
        elif list[i] > list[1]:
            b =list[i]

    return a, b

def FindList3MaxNum(list):

    max1, max2, max3 = None, None, None
    for num in list:
        if max1 is None or max1 < num:
            max1, num = num, max1
        if num is None:
            continue
        if max2 is None or num > max2:
            max2, num = num, max2
        if num is None:
            continue
        if max3 is None or num > max3:
            max3 = num

    return max1, max2, max3

def comput_bm25(bag, query_str):

    corpus = []
    for i in range(len(bag)):
        corpus.append(tokenization(bag[i]))
    bm25Model = bm25.BM25(corpus)

    query = []
    query_list = query_str.strip().split ()
    query_list = cut_stopwords (query_list)
    for word in query_list:
        query.append (word)

    scores = bm25Model.get_scores(query) # average_idf
    # scores.sort(reverse=True)
    ranked_scores = sorted (enumerate (scores), key=lambda x: x[1], reverse=True)
    print(ranked_scores)
    best_sent = []
    score_list = []
    for rank_id, (s_id, s_prob) in enumerate (ranked_scores):
        best_sent.append(bag[s_id])
        score_list.append(s_prob)
    # select top-k sentences as input
    # if len(scores)>1:
    #     index = FindList2MaxNum (scores)
    #     # Top-2
    #     best_sent = []
    #     for i in index:
    #         idx = scores.index(i)
    #         # idx_1 = scores.index (max (scores))
    #         best_sent.append(bag[idx])
    # else:
    #     idx_1 = scores.index (max (scores))
    #     best_sent = [bag[idx_1],' ']
    print(best_sent, score_list)
    return best_sent, score_list


def read_jsonl(input_file):
    "Read a jsonl file"
    lines = []
    with open (input_file, mode='r') as json_file:
        reader = jsonlines.Reader (json_file)
        for instance in reader:
            lines.append (instance)
    return lines

def write_jsonl(lines, output_file):
    "Write a jsonl file"
    # lines = []
    with jsonlines.open (output_file, mode='w') as writer:
        # writer.write()
        for instance in lines:
            writer.write({'article':instance.article,'article_ranked':instance.article_ranked,'question_answer':instance.choices, 'question':instance.query, 'ranked_scores':instance.scores, 'label':instance.label})

class InputExample(object):

    def __init__(self, guid, article, article_ranked, query, scores, choices=None, label=None):
        self.guid = guid
        self.article = article
        self.article_ranked = article_ranked
        self.query = query
        self.scores = scores
        self.choices = choices  # list in order
        self.label = label


def extract_ids(records, set_type='train'):
    examples = []

    for (i, line) in enumerate(records):
        record = line
        ex_id = str (i)
        guid = "%s-%s" % (set_type, ex_id)
        article = record['article']
        query = record['question']
        choices = record['question_answer']
        label = record['label']
        article_ranked, scores = comput_bm25(article,query)

        examples.append (
            InputExample(
                guid=guid,
                article=article,
                article_ranked=article_ranked,
                query=query,
                scores=scores,
                choices=choices,
                label=label
            )
        )
    return examples

def io_func(input_file, output_file):

    train_data = read_jsonl (input_file)
    write_jsonl (extract_ids(train_data), output_file)

def main():

    data_dir = "/data/semeval21_task4/SemEval2021-Task4"

    # input_file_train = os.path.join(data_dir, "training_data/Task_1_train_split.jsonl")
    # input_file_trail = os.path.join(data_dir, "trail_data/Task_1_Imperceptibility_split.jsonl")
    # input_file_dev = os.path.join(data_dir, "training_data/Task_1_dev_split.jsonl")
    #
    # output_file_train = os.path.join (data_dir, "training_data/Task_1_train_bm25.jsonl")
    # output_file_trail = os.path.join (data_dir, "trail_data/Task_1_Imperceptibility_bm25.jsonl")
    # output_file_dev = os.path.join (data_dir, "training_data/Task_1_dev_bm25.jsonl")

    input_file_train = os.path.join (data_dir, "training_data/Task_2_train_split.jsonl")
    input_file_trail = os.path.join (data_dir, "trail_data/Task_2_Nonspecificity_split.jsonl")
    input_file_dev = os.path.join (data_dir, "training_data/Task_2_dev_split.jsonl")
    input_file_trail_3 = os.path.join (data_dir, "trail_data/Task_3_Intersection_split.jsonl")

    output_file_train = os.path.join (data_dir, "training_data/Task_2_train_bm25.jsonl")
    output_file_trail = os.path.join (data_dir, "trail_data/Task_2_Nonspecificity_bm25.jsonl")
    output_file_dev = os.path.join (data_dir, "training_data/Task_2_dev_bm25.jsonl")
    output_file_trail_3 = os.path.join (data_dir, "trail_data/Task_3_Intersection_bm25.jsonl")

    io_func (input_file_train, output_file_train)
    io_func (input_file_trail, output_file_trail)
    io_func (input_file_dev, output_file_dev)
    io_func (input_file_trail_3, output_file_trail_3)



if __name__ == '__main__':
    main ()
