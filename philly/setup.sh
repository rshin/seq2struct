#!/bin/bash

pip install --user -e .
export PT_OUTPUT_DIR="${PT_OUTPUT_DIR:-$PWD}"
export CORENLP_HOME="$PT_DATA_DIR/third_party/stanford-corenlp-full-2018-10-05/"
