import argparse
import json

import _jsonnet
import attr

from seq2struct.utils import evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--section', required=True)
    parser.add_argument('--inferred', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    
    metrics = evaluation.compute_metrics(args.config, args.config_args, args.section, args.inferred)
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f)
        print('Wrote eval results to {}'.format(args.output))
    else:
        print(metrics)


if __name__ == '__main__':
    main()
