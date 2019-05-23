import argparse
import json
import os

import _jsonnet
import tqdm

from seq2struct import datasets
from seq2struct import models
from seq2struct.utils import registry
from seq2struct.utils import vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'])

    for section in config['data']:
        data = registry.construct('dataset', config['data'][section])
        for item in tqdm.tqdm(data, desc=section, dynamic_ncols=True):
            to_add, validation_info = model_preproc.validate_item(item, section)
            if to_add:
                model_preproc.add_item(item, section, validation_info)
    model_preproc.save()


if __name__ == '__main__':
    main()
