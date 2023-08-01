seeds=(42 43 44 45 46)
cudas=(0 2 4 5 6)

# model_name_or_path='bert-base-chinese'
model_name_or_path='hfl/chinese-roberta-wwm-ext-large'

for idx in ${!seeds[@]}; do
    nohup python -u eval_one_by_one.py \
    --model_name_or_path ${model_name_or_path} \
    --seed ${seeds[idx]} \
    --cuda ${cudas[idx]} \
    > ./log/eval_one_by_one_${model_name_or_path/\//-}_${seeds[idx]}.log &
done