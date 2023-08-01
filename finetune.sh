seeds=(42 43 44 45 46)
cudas=(0 2 4 5 6)

model_name_or_path='bert-base-chinese'
# model_name_or_path='hfl/chinese-roberta-wwm-ext-large'

for idx in ${!seeds[@]}; do
    nohup python -u finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --seed ${seeds[idx]} \
    --cuda ${cudas[idx]} \
    > ./log/finetune_weighted_${model_name_or_path/\//-}_${seeds[idx]}.log &
    sleep 3
done