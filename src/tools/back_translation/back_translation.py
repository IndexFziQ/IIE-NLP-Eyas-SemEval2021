#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : back_translation.py
@Author : Yuqiang Xie
@Date   : 2021/1/5
@E-Mail : indexfziq@gmail.com
"""
import requests
import json
import execjs  # pip install PyExecJS
from urllib.parse import quote
from tqdm import tqdm
import jsonlines

import datetime, time
import threading

# whether to print the logging
debug = False

class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        """
        why:
        Because the threading class has no return value,
        the MyThread class is redefined here so that the thread has the return value
        func. from https://www.cnblogs.com/hujq1029/p/7219163.html?utm_source=itdadao&utm_medium=referral
        """
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        # recieve the return value
        self.result = self.func(*self.args)

    def get_result(self):
        # if thread is on, return value is None
        try:
            return self.result
        except Exception:
            return None


# In order to limit the real request time or function execution time decorator
def limit_decor(limit_time):
    """
        :param limit_time:  set the maximum allowable execution time, unit: second
        :return:            return the decorated function return value without timeout, or None when timeout
    """

    def functions(func):
        # perform operation
        def run(*params):
            thre_func = MyThread(target=func, args=params)
            # When the main thread ends (exceeds the duration), the thread method ends
            thre_func.setDaemon(True)
            thre_func.start()
            # Count the number of sleeps in segments
            sleep_num = int(limit_time // 1)
            sleep_nums = round(limit_time % 1, 1)
            # Sleep briefly and try to get the return value many times
            for i in range(sleep_num):
                time.sleep(1)
                infor = thre_func.get_result()
                if infor:
                    return infor
            time.sleep(sleep_nums)
            # The final return value (regardless of whether the thread has ended)
            if thre_func.get_result():
                return thre_func.get_result()
            else:
                return"###request timeout###"  # Timeout return can be customized

        return run

    return functions

# Interface Function
def a1():
    print("###start request interface###")

    # Here the logic is encapsulated into a function, using thread calls
    a_theadiing = MyThread(target=a2)
    a_theadiing.start()
    a_theadiing.join()

    # return result
    a = a_theadiing.get_result()

    print("###request completed###")
    return a

@limit_decor(5)   #Timeout is set to 5s -- 5s The logic has not been executed and the return interface timeout

def a2(case):
    print("###start execution###")
    article_org = case['article']
    question_org = case['question']
    article_bk = back_translate(article_org)
    question_bk = back_translate(question_org)

    question_bk = question_bk.replace('@ placeholder', '@placeholder')
    case['bk_article'] = article_bk
    case['bk_question'] = question_bk
    print("###execution completed###")
    a=2
    return a


class Py4Js:

    def __init__(self):
        self.ctx = execjs.compile("""
            function TL(a) {
                var k = "";
                var b = 406644;
                var b1 = 3293161072;
                var jd = ".";
                var $b = "+-a^+6";
                var Zb = "+-3^+b+-f";
                for (var e = [], f = 0, g = 0; g < a.length; g++) {
                    var m = a.charCodeAt(g);
                    128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023),
                    e[f++] = m >> 18 | 240,
                    e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224,
                    e[f++] = m >> 6 & 63 | 128),
                    e[f++] = m & 63 | 128)
                }
                a = b;
                for (f = 0; f < e.length; f++) a += e[f],
                a = RL(a, $b);
                a = RL(a, Zb);
                a ^= b1 || 0;
                0 > a && (a = (a & 2147483647) + 2147483648);
                a %= 1E6;
                return a.toString() + jd + (a ^ b)
            };
            function RL(a, b) {
                var t = "a";
                var Yb = "+";
                for (var c = 0; c < b.length - 2; c += 3) {
                    var d = b.charAt(c + 2),
                    d = d >= t ? d.charCodeAt(0) - 87 : Number(d),
                    d = b.charAt(c + 1) == Yb ? a >>> d: a << d;
                    a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d
                }
                return a
            }
        """)

    def get_tk(self, text):
        return self.ctx.call("TL", text)


def build_url(text, tk, tl='zh-CN'):
    """
    URLEncoder
    :param text:
    :param tk:
    :param tl:
    :return:
    """
    return 'https://translate.google.cn/translate_a/single?client=webapp&sl=auto&tl=' + tl + '&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&source=btn&ssel=0&tsel=0&kc=0&tk=' \
           + str(tk) + '&q=' + quote(text, encoding='utf-8')


def translate(js, text, tl='zh-CN'):
    """
    tl -- the target language
    such as: fr, de, zh, ja, ko, ar
    """

    header = {
        'authority': 'translate.google.cn',
        'method': 'GET',
        'path': '',
        'scheme': 'https',
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh-CN,zh;q=0.9,ja;q=0.8',
        # 'cookie': '_ga=GA1.3.110668007.1547438795; _gid=GA1.3.791931751.1548053917; 1P_JAR=2019-1-23-1; NID=156=biJbQQ3j2gPAJVBfdgBjWHjpC5m9vPqwJ6n6gxTvY8n1eyM8LY5tkYDRsYvacEnWNtMh3ux0-lUJr439QFquSoqEIByw7al6n_yrHqhFNnb5fKyIWMewmqoOJ2fyNaZWrCwl7MA8P_qqPDM5uRIm9SAc5ybSGZijsjalN8YDkxQ',
         'cookie':'_ga=GA1.3.110668007.1547438795; _gid=GA1.3.1522575542.1548327032; 1P_JAR=2019-1-24-10; NID=156=ELGmtJHel1YG9Q3RxRI4HTgAc3l1n7Y6PAxGwvecTJDJ2ScgW2p-CXdvh88XFb9dTbYEBkoayWb-2vjJbB-Rhf6auRj-M-2QRUKdZG04lt7ybh8GgffGtepoA4oPN9OO9TeAoWDY0HJHDWCUwCpYzlaQK-gKCh5aVC4HVMeoppI',
        # 'cookie': '',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64)  AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36',
        'x-client-data': 'CKi1yQEIhrbJAQijtskBCMG2yQEIqZ3KAQioo8oBCL+nygEI7KfKAQjiqMoBGPmlygE='
    }
    url = build_url(text, js.get_tk(text), tl)
    res = []
    try:
        r = requests.get(url, headers=header)
        result = json.loads(r.text)
        r.encoding = "UTF-8"
        if debug:
            print(r.url)
            print(r.headers)
            print(r.request.headers)
            print(result)

        res = result[0]
        if res is None:
            if result[7] is not None:
                # If we enter the text incorrectly,
                # it prompts you whether you are looking for xxx,
                # then retranslate xxx and return
                try:
                    correct_text = result[7][0].replace('<b><i>', ' ').replace('</i></b>', '')
                    if debug:
                        print(correct_text)
                    correct_url = build_url(correct_text, js.get_tk(correct_text), tl)
                    correct_response = requests.get(correct_url)
                    correct_result = json.loads(correct_response.text)
                    res = correct_result[0]
                except Exception as e:
                    if debug:
                        print(e)
                    res = []

    except Exception as e:
        res = []
        if debug:
            print(url)
            print("translate" + text + "failed")
            print("Info of failure:")
            print(e)
    finally:
        return res


def get_translate(word, tl):
    js = Py4Js()
    translate_result = translate(js, word, tl)

    if debug:
        print("word== %s, tl== %s" % (word, tl))
        print(translate_result)
    return translate_result


def read_jsonl(input_file):
    "Read a jsonl file"
    lines = []
    with open(input_file, mode='r') as json_file:
        reader = jsonlines.Reader(json_file)
        for instance in reader:
            lines.append(instance)
    return lines


def back_translate(translate_text):

    print("source_translate_text:\n" + translate_text)

    results = get_translate(translate_text, 'fr')

    translate_result = []
    for result in results:
        if result[0] is None:
            break
        else:
            translate_result.append(result[0])

    bk_results = get_translate(' '.join(translate_result), 'en')

    bk_translate_result = []
    for bk_result in bk_results:
        if bk_result[0] is None:
            break
        else:
            bk_translate_result.append(bk_result[0])

    print("back_translate_result:\n" + ' '.join(bk_translate_result))
    return ' '.join(bk_translate_result)


# Program entry returns the value of a without timeout, returns the request timeout after timeout
if __name__ == '__main__':
    a = a1()  #Call the interface (here, the function a1 is regarded as an interface)
    input_file = '/data/training_data/Task_1_dev.jsonl'
    output_file = '/data/training_data/Task_1_dev_bktrans_google.jsonl'
    cases = read_jsonl(input_file)
    count = 0
    for i,case in tqdm(enumerate(cases)):
        a2(case)
    with open(output_file, 'w', encoding="utf-8") as fout:
        for feature in cases:
            if feature['bk_article']!="":
                feature['article'] = feature['bk_article']
                if feature['bk_question']!="":
                    feature['question'] = feature['bk_question']
                    fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
                else:
                    fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
            else:
                if feature['bk_question']!="":
                    feature['question'] = feature['bk_question']
                    fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
                else:
                    continue


