#!/bin/bash

#clear
set -e
cd ..

dset=wikipedia
nc=10
d_txt=300
# ALPHA=(0 0.0001 0.001 0.01 0.1 1 10)
# BETA=(0 0.0001 0.001 0.01 0.1 1 10)
# GAMMA=(0 0.0001 0.001 0.01 0.1 1 10)
ALPHA=(0.01 0.1 1)
BETA=(0.1 1 10)
GAMMA=(0.0001 0.001 0.01 0.1 1 10)


# echo "gamma = (10 0 0 10 0)"
# echo "alpha = (1 10 10 10 10)"
# echo "beta = (0.001 0.0001 0 0.001 0.1)"
# echo "epoch = (6 5 7 15 ?)"

echo grid search
for k in $(seq 0 4); do
    if [ $k -ne 0 ]; then continue; fi
    for _g in $(seq 0 `expr ${#GAMMA[@]} - 1`); do
        gamma=${GAMMA[_g]}
        if [ $gamma != "10" ]; then continue; fi
        for _a in $(seq 0 `expr ${#ALPHA[@]} - 1`); do
            alpha=${ALPHA[_a]}
            for _b in $(seq 0 `expr ${#BETA[@]} - 1`); do
                beta=${BETA[_b]}
                echo grid-search $dset, split: $k-DADN, "LRY's tuning mode"
                echo alpha: $alpha, beta: $beta, gamma: $gamma
                python main.py --donot_save_model \
                    --dataset $dset --data_path data/$dset \
                    --split_id $k --split_file data/$dset/disjoint/split-$k-DADN/split.$dset.$k.DADN.mat \
                    --n_class $nc --sparse_label --dim_image 4096 --dim_text $d_txt \
                    --alpha $alpha --beta $beta --gamma $gamma \
                    --tune 2 --log_path log/zero-shot/tune_a${alpha}_b${beta}_g${gamma}
                # break
            done
            # break
        done
        # break
    done
    # break
done

# echo tune gamma
# for k in $(seq 0 4); do
#     if [ $k -ne 0 ]; then continue; fi
#     for _g in $(seq 0 `expr ${#GAMMA[@]} - 1`); do
#         gamma=${GAMMA[_g]}
#         # if [ $gamma != "10" ]; then continue; fi
#         echo tune $dset, split: $k-DADN, gamma: $gamma, "LRY's tuning mode"
#         python main.py --donot_save_model \
#             --dataset $dset --data_path data/$dset \
#             --split_id $k --split_file data/$dset/disjoint/split-$k-DADN/split.$dset.$k.DADN.mat \
#             --n_class $nc --sparse_label --dim_image 4096 --dim_text $d_txt \
#             --alpha 0.01 --beta 1 --gamma $gamma \
#             --tune 2 --log_path log/zero-shot/tune_g$gamma
#         # break
#     done
#     # break
# done

# echo tune alpha
# gamma_ls=(10 0 0 10 0)
# for k in $(seq 0 4); do
#     if [ $k -ne 4 ]; then continue; fi
#     gamma=${gamma_ls[k]}
#     for _a in $(seq 0 `expr ${#ALPHA[@]} - 1`); do
#         alpha=${ALPHA[_a]}
#         # if [ $alpha != "10" ]; then continue; fi
#         echo tune $dset on split-$k-DADN, alpha: $alpha, gamma: $gamma, "LRY's tuning mode"
#         python main.py --donot_save_model \
#             --dataset $dset --data_path data/$dset \
#             --split_id $k --split_file data/$dset/disjoint/split-$k-DADN/split.$dset.$k.DADN.mat \
#             --n_class $nc --sparse_label --dim_image 4096 --dim_text $d_txt \
#             --alpha $alpha --beta 1 --gamma $gamma \
#             --tune 2 --log_path log/zero-shot/tune_a$alpha
#         # break
#     done
# done

# echo tune beta
# gamma_ls=(10 0 0 10 0)
# alpha_ls=(1 10 10 10 10)
# for k in $(seq 0 4); do
#     if [ $k -ne 0 ]; then continue; fi
#     gamma=${gamma_ls[k]}
#     alpha=${alpha_ls[k]}
#     for _b in $(seq 0 `expr ${#BETA[@]} - 1`); do
#         beta=${BETA[_b]}
#         # if [ $beta != "10" ]; then continue; fi
#         echo tune $dset on split-$k-DADN, alpha: $alpha, beta: $beta, gamma: $gamma, "LRY's tuning mode"
#         python main.py --donot_save_model \
#             --dataset $dset --data_path data/$dset \
#             --split_id $k --split_file data/$dset/disjoint/split-$k-DADN/split.$dset.$k.DADN.mat \
#             --n_class $nc --sparse_label --dim_image 4096 --dim_text $d_txt \
#             --alpha $alpha --beta $beta --gamma $gamma \
#             --tune 2 --log_path log/zero-shot/tune_b$beta
#         # break
#     done
# done

# echo tune epoch
# gamma_ls=(10 0 0 10 0)
# alpha_ls=(1 10 10 10 10)
# beta_ls=(0.001 0.0001 0 0.001 0.1)
# for k in $(seq 0 4); do
#     if [ $k -ne 4 ]; then continue; fi
#     gamma=${gamma_ls[k]}
#     alpha=${alpha_ls[k]}
#     beta=${beta_ls[k]}
#     echo tune $dset on split-$k-DADN, alpha: $alpha, beta: $beta, gamma: $gamma, "LRY's tuning mode"
#     python main.py --donot_save_model \
#         --dataset $dset --data_path data/$dset \
#         --split_id $k --split_file data/$dset/disjoint/split-$k-DADN/split.$dset.$k.DADN.mat \
#         --n_class $nc --sparse_label --dim_image 4096 --dim_text $d_txt \
#         --alpha $alpha --beta $beta --gamma $gamma \
#         --tune 2 --log_path log/zero-shot/tune_e
# done
