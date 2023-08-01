import os
import argparse
import logging

import torch
import evaluate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (BertConfig, BertTokenizer, default_data_collator,
                          DataCollatorWithPadding)
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from utils.dataloader import load_data
from utils.models import BertForSequenceClassification, RobertaForSequenceClassification

MODEL_CLASS_MAPPING = {
    'bert-base-chinese': BertForSequenceClassification,
    'hfl/chinese-roberta-wwm-ext-large': RobertaForSequenceClassification
}

logger = get_logger(__name__)


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
        required=True)
    parser.add_argument(
        '--per_device_test_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.')
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

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    datasets = load_data(['test'])

    label_list = (datasets['test']).unique('label')
    label_list.sort()
    num_labels = len(label_list)

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

    padding = 'max_length' if args.pad_to_max_length else False

    def preprocess_function(examples):
        # tokenize the texts
        context = examples['context']
        response = examples['response']
        result = tokenizer(text=context,
                           text_pair=response,
                           padding=padding,
                           max_length=args.max_length,
                           truncation=True,
                           add_special_tokens=True,
                           return_token_type_ids=True)
        result['labels'] = examples['label']
        return result

    with accelerator.main_process_first():
        process_datasets = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=['context', 'response', 'label'],
            desc='running tokenizer on dataset')

    test_dataset = process_datasets['test']

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=(8 if Accelerator.mixed_precision == 'fp16' else
                                None))

    test_dataloader = DataLoader(test_dataset,
                                 shuffle=True,
                                 collate_fn=data_collator,
                                 batch_size=args.per_device_test_batch_size)

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    acc_metric = evaluate.load('accuracy')
    test_preds = []
    test_trues = []

    model.eval()
    test_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            test_loss += loss.detach().float()
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather(
            (predictions, batch["labels"]))
        acc_metric.add_batch(predictions=predictions, references=references)

        test_preds.extend(list(predictions.detach().cpu().numpy()))
        test_trues.extend(list(references.detach().cpu().numpy()))

    eval_acc = acc_metric.compute()
    logger.info(f'accuracy: {eval_acc}')
    logger.info(f'validation loss: {test_loss}')

    report = classification_report(test_trues, test_preds, digits=5)
    logger.info(f'report: \n{report}')


if __name__ == '__main__':
    main()
