#!/bin/bash

source ./third_party/argparser/argparser.sh
parse_args "$@"

set -o errexit -o pipefail -o noclobber -o xtrace

source ./philly/setup.sh
if [[ "$preprocess" = true ]]; then
    python3 preprocess.py \
        --config configs/spider-idioms/nl2code-0518.jsonnet \
        --config-args "{bs: $bs, st: '$st', nt: $nt, lr: 2.5e-4, end_lr: 0, att: 0, data_path: '$PT_DATA_DIR/spider-20190205/'}"
fi
python3 train.py \
    --config configs/spider-idioms/nl2code-0518.jsonnet \
    --config-args "{bs: $bs, st: '$st', nt: $nt, lr: 2.5e-4, end_lr: 0, att: 0, data_path: '$PT_DATA_DIR/spider-20190205/'}" \
    --logdir "$PT_OUTPUT_DIR/logdirs/20190519/"
