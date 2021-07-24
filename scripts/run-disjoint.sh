#!/bin/bash

# clear
set -e
cd ..

dset=wikipedia
# gamma_ls=(10 0 0 10 0)
# alpha_ls=(1 10 10 10 10)
# beta_ls=(0.001 0.0001 0 0.001 0.1)
# epoch_ls=(6 5 7 15 ?)
# for k in $(seq 0 4); do
#     if [ $k -ne 0 ]; then continue; fi
#     alpha=${alpha_ls[k]}
#     beta=${beta_ls[k]}
#     gamma=${gamma_ls[k]}
#     epoch=${epoch_ls[k]}
#     echo $dset, split-$k
#     CUDA_VISIBLE_DEVICES=3,0,1,2 \
#     python main.py \
#         --dataset $dset --data_path data/$dset \
#         --split_id $k --split_file data/$dset/disjoint/split-$k-DADN/split.$dset.$k.DADN.mat \
#         --n_class 10 --sparse_label --dim_text 300 \
#         --alpha $alpha --beta $beta --gamma $gamma --epoch $epoch \
#         --log_path log/zero-shot
# done


echo "re-test the OK network"
k=0
python main.py --donot_save_model \
    --dataset $dset --data_path data/$dset \
    --split_id $k --split_file data/$dset/disjoint/split-$k-DADN/split.$dset.$k.DADN.mat \
    --n_class 10 --sparse_label --dim_text 300 \
    --alpha 0.01 --beta 0.1 --gamma 0.0001 --epoch 20 \
    --log_path log/zero-shot
