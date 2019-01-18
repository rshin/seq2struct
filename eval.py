import argparse
import json

import _jsonnet
import attr

from seq2struct.utils import evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--section', required=True)
    parser.add_argument('--inferred', required=True)
    args = parser.parse_args()
    
    print(evaluation.compute_metrics(args.config, args.section, args.inferred))


if __name__ == '__main__':
    main()
