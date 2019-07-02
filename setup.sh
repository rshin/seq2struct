#!/bin/bash

pip install --user -e .
sudo apt-get update
sudo apt-get install -y default-jre
export PT_OUTPUT_DIR="${PT_OUTPUT_DIR:-$PWD}"
export PT_DATA_DIR="${PT_DATA_DIR:-$PWD}"

export CORENLP_HOME="$PT_DATA_DIR/third_party/stanford-corenlp-full-2018-10-05/"
export CACHE_DIR=${PT_DATA_DIR}

export LC_ALL="C.UTF-8"
export LANG="C.UTF-8"
