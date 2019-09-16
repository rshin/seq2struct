# Merge outputs of infer.py and other models for comparison.
# Outputs to a CSV file.

import argparse
import csv
import json
import os

from third_party.spider import evaluation


def main():
    parser = argparse.ArgumentParser()
    # Outputs of infer.py
    parser.add_argument('--infer', nargs='*', default=())
    # Files containing inferred SQL, one per line
    # in order of the items in the dev set.
    parser.add_argument('--sql', nargs='*', default=())
    # The name to output for each of the inputs, in the CSV header.
    parser.add_argument('--names', nargs='*', default=())
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    assert len(args.names) == len(args.infer) + len(args.sql)

    SPIDER_ROOT = 'data/spider-20190205'
    foreign_key_maps = {
        db['db_id']: evaluation.build_foreign_key_map(db)
        for db in json.load(open(os.path.join(SPIDER_ROOT, 'tables.json')))
    }
  
    # 1. Create the evaluator
    evaluator = evaluation.Evaluator(
          os.path.join(SPIDER_ROOT, 'database'),
          foreign_key_maps,
          'match')

    # 2. Read the ground truth SQL
    dev = json.load(open(os.path.join(SPIDER_ROOT, 'dev.json')))

    # 3. Perform evaluation
    difficulty = {}
    inferred_per_file = []
    correct_per_file = []
    # db
    # gold

    for infer_path in args.infer:
        inferred = [None] * len(dev)
        correct = [None] * len(dev)
        inferred_per_file.append(inferred)
        correct_per_file.append(correct)

        for line in open(infer_path):
            item = json.loads(line)
            item_inferred = item['beams'][0]['inferred_code']
            i = item['index']

            eval_result = evaluator.evaluate_one(
                  db_name=dev[i]['db_id'],
                  gold=dev[i]['query'],
                  predicted=item_inferred)
            
            difficulty[i] = eval_result['hardness']
            inferred[i] = item_inferred
            correct[i] = 1 if eval_result['exact'] else 0

    for sql_path in args.sql:
        inferred = [None] * len(dev)
        correct = [None] * len(dev)
        inferred_per_file.append(inferred)
        correct_per_file.append(correct)

        for i, line in enumerate(open(sql_path)):
            eval_result = evaluator.evaluate_one(
                  db_name=dev[i]['db_id'],
                  gold=dev[i]['query'],
                  predicted=line.strip())
            
            difficulty[i] = eval_result['hardness']
            inferred[i] = line.strip()
            correct[i] = 1 if eval_result['exact'] else 0


    with open(args.out, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['DB', 'Difficulty', 'Question', 'Gold'] + ['{} correct'.format(c)
          for c in args.names] + ['{} output'.format(c) for c in args.names])

        for i, dev_item in enumerate(dev):
            writer.writerow(
                [dev_item['db_id'], difficulty[i], dev_item['question'],
                  dev_item['query']] +
                [x[i] for x in correct_per_file] +
                [x[i] for x in inferred_per_file])



if __name__ == '__main__':
    main()

