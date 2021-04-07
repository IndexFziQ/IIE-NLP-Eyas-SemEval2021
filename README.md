# SemEval2021 - Task 4: ReCAM: Reading Comprehension of Abstract Meaning @IIE-NLP-Eyas

ReCAM has three subtasks. Subtask 1 and 2 focus on evaluating machine learning models' performance with regard to two definitions of abstractness, which we call *imperceptibility* and *nonspecificity*, respectively. Subtask 3 aims to provide some insights to their relationships.

This repository contains **preliminary** code for the paper titled:

[IIE-NLP-Eyas at SemEval-2021 Task 4: Enhancing PLM for ReCAM with Special Tokens, Re-Ranking, Siamese Encoders and Back Translation](https://arxiv.org/abs/2102.12777). Yuqiang Xie, Luxi Xing, Wei Peng and Yue Hu*. SemEval 2021@ACL-IJCNLP2021.

## Data

**Data Format**

Data is stored one-question-per-line in json format. Each instance of the data can be trated as a python dictinoary object. See examples below for further help in reading the data.


**Sample**

```
{
"article": "... observers have even named it after him, ``Abenomics". It is based on three key pillars -- the "three arrows" of monetary policy, fiscal stimulus and structural reforms in order to ensure long-term sustainable growth in the world's third-largest economy. In this weekend's upper house elections, ....",
"question": "Abenomics: The @placeholder and the risks",
"option_0": "chances",
"option_1": "prospective",
"option_2": "security",
"option_3": "objectives",
"option_4": "threats",
"label": 3
}
```

* article : the article that provide the context for the question.
* question : the question models are required to answer.
* options : five answer options for the question. Model are required to select the true answer from 5 options.
* label : index of the answer in options

**Code**

We implemented our System based on HuggingFace [transformers](https://github.com/huggingface/transformers).  We are still cleaning up the code! Full code documentation will be ready soon!

#### Reference

Please cite our paper using the following bibtex:

```
@misc{xie2021iienlpeyas,
    title={IIE-NLP-Eyas at SemEval-2021 Task 4: Enhancing PLM for ReCAM with Special Tokens, Re-Ranking, Siamese Encoders and Back Translation},
    author={Yuqiang Xie and Luxi Xing and Wei Peng and Yue Hu},
    year={2021},
    eprint={2102.12777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

#### License

MIT

