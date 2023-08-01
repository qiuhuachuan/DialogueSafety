# 🎉 Dialogue Safety in Mental Health Support

<a href='https://arxiv.org/abs/2307.16457'><img src='https://img.shields.io/badge/PaperLink-ArXiv 2307.16457-red'></a>

This repository contains the code and data for the paper titled `A Benchmark for Understanding Dialogue Safety in Mental Health Support`, which is accepted to **The 12th CCF International Conference on Natural Language Processing and Chinese Computing (NLPCC2023)**.

## 📗 Taxonomy and Framework

![](./image/framework.png)
Figure 1. Our proposed sequential and inclusive taxonomy aims to ensure safe and supportive dialogue for mental health support. In a given conversational context, annotators are required to sequentially label the model responses based on the node order. It is crucial to note that content that is not understandable will be considered unsafe. Therefore, addressing this issue takes precedence in our sequential framework, which occurs in Node 1. The four rightmost categories in Node 4 cover the unsafe classes in the existing chitchat scenarios.

![](./image/examples.png)
Table 1. Different types of model responses within a dialogue context in our taxonomy.

## 🔥 Results

![](./image/results.png)
Table 2. Evaluation results for fine-grained classification on the test set. The results present the mean and standard deviation (subscript) of accuracy (Acc.), precision (P), recall (R), and F1 score (F1). In the few-shot setting, the inference prompt includes 8 diverse examples from the training set. † indicates that the model used is GPT-3.5-TURBO-0301, while ‡ indicates that the model used is GPT-3.5-TURBO-0613.

## 🌟 data

Due to privacy concerns regarding psychological counseling data, we currently only release the **test set**.
The data used in this paper is included in the `data` directory. The data has been appropriately anonymized.

Access to our **training set** is restricted to researchers who agree to comply with ethical guidelines and sign a confidentiality agreement with us. If you need access to our dataset, please sign the data use agreement in `DataUseAgreement` and contact Huachuan (`qiuhuachuan@westlake.edu.cn`).

Below is the label description.

```Python
{
    0: "Nonsense",
    1: "Humanoid Mimicry",
    2: "Linguistic Neglect",
    3: "Unamiable Judgment",
    4: "Toxic Language",
    5: "Unauthorized Preachment",
    6: "Nonfactual Statement",
    7: "Safe Response"
}
```

## 🔍 ChatGPT Eval

`ChatGPT_eval.py` is used to generate model responses in a zero- and few-shot paradigm. For analysis, please execute the script `analyze_ChatGPT_results.py`.

## 🛠 Dependencies

The code is implemented using Python 3.10 and PyTorch v2.0. We recommend using Anaconda or Miniconda to set up this codebase. Please install dependencies through requirements.txt.

```Bash
pip install -r requirements.txt
```

## 🎯 Training

The code includes two models: bert-base-chinese and hfl/chinese-roberta-wwm-ext-large. Users can choose either one based on their requirements or opt for other models available on Hugging Face.

```Bash
bash finetune.sh
```

## 🎨 Evaluation

The code includes two models: bert-base-chinese and hfl/chinese-roberta-wwm-ext-large.

```Bash
bash eval.sh
```

## 📚 BibTeX entry and citation info

If you find the data or paper useful, kindly cite it in your own work.

```bibtex
@misc{qiu2023benchmark,
      title={A Benchmark for Understanding Dialogue Safety in Mental Health Support},
      author={Huachuan Qiu and Tong Zhao and Anqi Li and Shuai Zhang and Hongliang He and Zhenzhong Lan},
      year={2023},
      eprint={2307.16457},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
