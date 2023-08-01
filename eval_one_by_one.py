import os
import argparse
import logging
import ujson

import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import BertConfig, BertTokenizer
from sklearn.metrics import classification_report
from torch.nn.functional import softmax

from utils.models import BertForSequenceClassification, RobertaForSequenceClassification

MODEL_CLASS_MAPPING = {
    'bert-base-chinese': BertForSequenceClassification,
    'hfl/chinese-roberta-wwm-ext-large': RobertaForSequenceClassification
}

logger = get_logger(__name__)

label_mapping = {
    0: 'Nonsense',
    1: 'Humanoid Mimicry',
    2: 'Linguistic Neglect',
    3: 'Unamiable Judgment',
    4: 'Toxic Language',
    5: 'Unauthorized Preachment',
    6: 'Nonfactual Statement',
    7: 'Safe Response'
}


def parse_args():
    parser = argparse.ArgumentParser(
        'Finetune a transformers model on a text classification task.')
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help=
        ('The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,'
         ' sequences shorter will be padded if `--pad_to_max_length` is passed.'
         ))
    parser.add_argument(
        '--pad_to_max_length',
        action='store_true',
        help=
        'If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help=
        'Path to pretrained model or model identifier from huggingface.co/models.',
        default='bert-base-chinese')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='A seed for reproducible training.')

    parser.add_argument('--cuda', type=str, default='0', help='cuda device')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    num_labels = 8

    config = BertConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=num_labels,
                                        finetuning_task='text classification')
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        never_split=['[seeker]', '[supporter]'])
    tokenizer.vocab['[seeker]'] = tokenizer.vocab.pop('[unused1]')
    tokenizer.vocab['[supporter]'] = tokenizer.vocab.pop('[unused2]')

    MODEL_CLASS = MODEL_CLASS_MAPPING[args.model_name_or_path]

    model = MODEL_CLASS(config, args.model_name_or_path)
    model_name_or_path: str = args.model_name_or_path
    model_name_or_path = model_name_or_path.replace('/', '-')
    PATH = f'out/{model_name_or_path}/{args.seed}/pytorch_model.bin'
    model.load_state_dict(torch.load(PATH))
    model.cuda()

    padding = 'max_length' if args.pad_to_max_length else False
    model.eval()

    with open('./data/test.json', 'r', encoding='utf-8') as f:
        test_data = ujson.load(f)

    target_dir = f'./finetuned_model_predict/{model_name_or_path}/{args.seed}'
    os.makedirs(target_dir, exist_ok=True)
    existings = os.listdir(target_dir)

    for idx, item in enumerate(test_data):

        if f'{idx}.json' in existings:
            print(f'{idx}.json DONE')
        else:
            context = item['context']
            response = item['response']

            result = tokenizer.encode_plus(text=context,
                                           text_pair=response,
                                           padding=padding,
                                           max_length=args.max_length,
                                           truncation=True,
                                           add_special_tokens=True,
                                           return_token_type_ids=True,
                                           return_tensors='pt')
            result = result.to('cuda')

            with torch.no_grad():
                outputs = model(**result)
                predictions = outputs.logits.argmax(dim=-1)
                pred_label = predictions.item()

                item['predict_label'] = pred_label

                with open(f'{target_dir}/{idx}.json', 'w',
                          encoding='utf-8') as f:
                    ujson.dump(item, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
    print('done')