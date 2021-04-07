#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : compare_dup_data_t4sta.py
@Time    : 2020-11-11 10:23
@Author  : Yuqiang Xie
@Contact : indexfziq@gmail.com
"""
# Used for compare the dup data between train data set
# and trail/dev data set in sub-task A

import jsonlines
import os
import random

data_dir = "/data/semeval21_task4/SemEval2021-Task4"

input_file_train = os.path.join(data_dir, "training_data/Task_1_train.jsonl")
input_file_trail = os.path.join(data_dir, "trail_data/Task_1_Imperceptibility.jsonl")
input_file_dev = os.path.join(data_dir, "training_data/Task_1_dev.jsonl")

def read_jsonl(input_file):
    "Read a jsonl file"
    lines = []
    with open (input_file, mode='r') as json_file:
        reader = jsonlines.Reader (json_file)
        for instance in reader:
            lines.append (instance)
    return lines


class InputExample(object):

    def __init__(self, guid, question, choices=None, label=None):
        self.guid = guid
        self.question = question
        self.choices = choices  # list in order
        self.label = label

def extract_ids(records, set_type='train'):
    examples = []

    for (i, line) in enumerate (records):
        record = line
        ex_id = str (i)
        guid = "%s-%s" % (set_type, ex_id)
        article = record['article']
        question = record['question']

        opt1 = record['option_0']
        opt1 = replace_placeholder (question, opt1)
        opt2 = record['option_1']
        opt2 = replace_placeholder (question, opt2)
        opt3 = record['option_2']
        opt3 = replace_placeholder (question, opt3)
        opt4 = record['option_3']
        opt4 = replace_placeholder (question, opt4)
        opt5 = record['option_4']
        opt5 = replace_placeholder (question, opt5)
        label = record['label']

        examples.append (
            InputExample(
                guid=guid,
                question=article,
                choices=[opt1, opt2, opt3, opt4, opt5],
                label=label
            )
        )
    return examples

# replace placeholder of question with options
def replace_placeholder(str, opt):
    list = str.split(' ')
    for i in range(len(list)):
        if list[i] == '@placeholder':
            list[i] = opt
    final_opt = ' '.join(list)
    return final_opt

def compare_two(a,b):
    if a.question==b.question:
        return a.question+'\n'+'\n'.join(a.choices)
    else:
        return '0'

def output(file_train, file_comp, type='dev'):
    print ("-" * 3 + 'compare duplicate data between train and ' + type + "-" * 3)
    train_data = read_jsonl (file_train)
    comp_data = read_jsonl (file_comp)

    examples_train = extract_ids (train_data)
    examples_comp = extract_ids (comp_data)

    repeat_ids=[]
    for (i,linei) in enumerate(examples_train):
        for (j,linej) in enumerate(examples_comp):
            example_ = compare_two(linei,linej)
            if example_!='0':
                if example_==None:
                    continue
                else:
                    repeat_ids.append(example_)

    print(type+'\t'+'Repetition Example:\n%s' % (repeat_ids[random.randint(1,len(repeat_ids))]))
    print(type+'\t'+'Repetition Num:\t%s\t Example Num Total:\t%s' % (len(repeat_ids),len(examples_comp)))
    print(type+'\t'+'Repetition Rate:\t%s' % (len(repeat_ids)/len(examples_comp)))

def main():
    output(input_file_train, input_file_trail, type='trail')
    output(input_file_train, input_file_dev, type='dev')

if __name__ == '__main__':
    main ()

