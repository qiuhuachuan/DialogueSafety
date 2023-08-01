import ujson

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


def find_key_with_max_value(d):
    max_value = max(d.values())
    for k, v in d.items():
        if v == max_value:
            return k


# 分析BERT 和 RoBERTa
def cal_for_finetuned(model: str):
    total = 0
    model_counter = 0
    for idx in range(800):
        eval_dict = {'true': {}, 'predict': {}}
        for test_round in [42, 43, 44, 45, 46]:
            target_dir = f'./finetuned_model_predict/{model}/{test_round}'
            with open(f'{target_dir}/{idx}.json', 'r', encoding='utf-8') as f:
                data = ujson.load(f)
            label = data['label']
            predict_label = data['predict_label']
            if label == 0:
                if label in eval_dict['true']:
                    eval_dict['true'][label] += 1
                else:
                    eval_dict['true'][label] = 1
                if predict_label in eval_dict['predict']:
                    eval_dict['predict'][predict_label] += 1
                else:
                    eval_dict['predict'][predict_label] = 1

        if 0 in eval_dict['true']:
            # print(f'{target_dir}/{idx}.json')
            # print(eval_dict)
            total += 1
            pass
        if 0 in eval_dict['predict']:
            model_counter += 1

    return total, model_counter


# 分析GPT-3.5-0613-few-shot
def cal_for_GPT(model_version: str, shot_type: str):
    gpt_counter = 0
    for idx in range(800):
        eval_dict = {'true': {}, 'predict': {}}
        for test_round in ['one', 'two', 'three', 'four', 'five']:
            target_dir = f'./ChatGPT_gen/{model_version}/{shot_type}/{test_round}'
            with open(f'{target_dir}/{idx}.json', 'r', encoding='utf-8') as f:
                data = ujson.load(f)
            label = data['item']['label']
            predict_str = data['choices'][0]['message']['content']
            if '.' in predict_str:
                predict_str = predict_str.split('.')[0]

            if predict_str in description_mapping:
                predict_label = description_mapping[predict_str]
            if label == 0:
                if label in eval_dict['true']:
                    eval_dict['true'][label] += 1
                else:
                    eval_dict['true'][label] = 1
                if predict_label in eval_dict['predict']:
                    eval_dict['predict'][predict_label] += 1
                else:
                    eval_dict['predict'][predict_label] = 1

        if 0 in eval_dict['true']:
            # print(f'{target_dir}/{idx}.json')
            # print(eval_dict)
            pass
        if 0 in eval_dict['predict']:
            gpt_counter += 1
    return gpt_counter


total, bert_counter = cal_for_finetuned(model='bert-base-chinese')
total, roberta_counter = cal_for_finetuned(
    model='hfl-chinese-roberta-wwm-ext-large')
gpt_0301_zero_shot_counter = cal_for_GPT(model_version='0301',
                                         shot_type='zero_shot')
gpt_0301_few_shot_counter = cal_for_GPT(model_version='0301',
                                        shot_type='few_shot')
gpt_0613_zero_shot_counter = cal_for_GPT(model_version='0613',
                                         shot_type='zero_shot')
gpt_0613_few_shot_counter = cal_for_GPT(model_version='0613',
                                        shot_type='few_shot')
print('total:', total)

print('gpt_0301_zero_shot_counter:', gpt_0301_zero_shot_counter)
print('gpt_0301_few_shot_counter:', gpt_0301_few_shot_counter)
print('gpt_0613_zero_shot_counter:', gpt_0613_zero_shot_counter)
print('gpt_0613_few_shot_counter:', gpt_0613_few_shot_counter)
print('bert counter:', bert_counter)
print('roberta counter:', roberta_counter)
