import argparse
import json
import os

import _jsonnet

import seq2struct
from seq2struct.utils import registry
from seq2struct.utils import vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = json.loads(_jsonnet.evaluate_file(args.config))

    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'])

    train_data = registry.construct('dataset', config['data']['train'])
    val_data = registry.construct('dataset', config['data']['val'])

    for section, data in (('train', train_data), ('val', val_data)):
        for item in data:
            to_add, validation_info = model_preproc.validate_item(item, section)
            if to_add:
                model_preproc.add_item(item, section, validation_info)
        model_preproc.save()


if __name__ == '__main__':
    main()