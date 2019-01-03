import argparse
import ast
import collections
import copy
import datetime
import itertools
import json
import os

import _jsonnet
import attr
import asdl
import astor
import torch
import tqdm

from seq2struct import ast_util
from seq2struct import beam_search
from seq2struct.utils import registry
from seq2struct.utils import saver as saver_mod
from seq2struct.utils import vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--inferred', required=True)
    args = parser.parse_args()
    config = json.loads(_jsonnet.evaluate_file(args.config))

    inferred = open(args.inferred)

    metrics = registry.lookup('dataset', config['data']['val']).Metrics()
    for line in inferred:
        infer_results = json.loads(line)
        metrics.add(infer_results['gold_code'], infer_results['beams'][0]['inferred_code'])
    print(metrics.finalize())


if __name__ == '__main__':
    main()
