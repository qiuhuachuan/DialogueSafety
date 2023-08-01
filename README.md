# Dialogue Safety in Mental Health Support

This repository contains the code and data for the paper titled "A Benchmark for Understanding Dialogue Safety in Mental Health Support," which is accepted to **The 12th CCF International Conference on Natural Language Processing and Chinese Computing (NLPCC2023)**.

## data

The data used in this paper is included in the `data` directory. The data has been appropriately anonymized.

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

## Dependencies

The code is implemented using Python 3.10 and PyTorch v2.0. We recommend using Anaconda or Miniconda to set up this codebase. Please install dependencies through requirements.txt.

```Bash
pip install -r requirements.txt
```

## Training

The code includes two models: bert-base-chinese and hfl/chinese-roberta-wwm-ext-large. Users can choose either one based on their requirements or opt for other models available on Hugging Face.

```Bash
bash finetune.sh
```

## Evaluation

The code includes two models: bert-base-chinese and hfl/chinese-roberta-wwm-ext-large.

```Bash
bash eval.sh
```

## BibTeX entry and citation info

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
