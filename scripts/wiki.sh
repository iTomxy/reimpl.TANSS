#!/bin/bash

# clear
set -e
cd ..

dset=wikipedia

python main.py --donot_save_model \
    --dataset $dset --sparse_label \
    --alpha 0.01 --beta 0.1 --gamma 0.0001 --epoch 20 --test_per 5
