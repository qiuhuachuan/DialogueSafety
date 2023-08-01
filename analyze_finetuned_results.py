import ujson
import argparse

from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    default='bert-base-chinese',
    type=str,
    choices=['bert-base-chinese', 'hfl/chinese-roberta-wwm-ext-large'])

args = parser.parse_args()

model = args.model.replace('/', '-')
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

description_mapping = {}
for idx in range(8):
    description_mapping[label_mapping[idx]] = idx

with open('./data/test.json', 'r', encoding='utf-8') as f:
    test_data = ujson.load(f)

for seed in [42, 43, 44, 45, 46]:
    print(seed)
    counter = 0
    y_preds = []
    y_trues = []
    bi_preds = []
    bi_trues = []
    for idx, item in enumerate(test_data):
        true_label = item['label']
        y_trues.append(true_label)
        if true_label == 7:
            bi_trues.append(1)
        else:
            bi_trues.append(0)

        current_file = f'./finetuned_model_predict/{model}/{seed}/{idx}.json'
        with open(current_file, 'r', encoding='utf-8') as f:
            data = ujson.load(f)
            predict_label = data['predict_label']
            y_preds.append(predict_label)

            counter += 1
            if predict_label == 7:
                bi_preds.append(1)
            else:
                bi_preds.append(0)

    print('counter:', counter)
    print('total:')
    print(classification_report(y_trues, y_preds, digits=5))
    print('binary:')
    print(classification_report(bi_trues, bi_preds, digits=5))
    print('matrix')
    print(confusion_matrix(bi_trues, bi_preds))