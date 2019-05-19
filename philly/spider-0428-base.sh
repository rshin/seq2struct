#!/bin/bash -x

source ./philly/setup.sh
python3 train.py \
    --config configs/spider-20190205/nl2code-0428-stability.philly.jsonnet \
    --config-args "{bs: 50, lr: 1e-3, end_lr: 0, att: 0, data_path: '$PT_DATA_DIR/spider-20190205/'}" \
    --logdir "$PT_OUTPUT_DIR/logdirs/20190428-stability/"
