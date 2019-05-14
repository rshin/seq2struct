import itertools
import os
import sys


def main():
    all_commands = []
    all_eval_commands = []

    for bs, lr, end_lr in (
        (50, 1e-3, 0),
        (100, 1e-3, 0),
        (10, 5e-4, 0),
        (10, 2.5e-4, 0),
        (10, 1e-3, 5e-4),
        (10, 1e-3, 2.5e-4),
    ):
        steps = list(range(1100, 40000, 1000)) + [40000]
        for step in steps:
            infer_command = ((
                'python infer.py --config configs/spider-20190205/nl2code-0428-stability.jsonnet ' +
                '--logdir logdirs/20190428-stability ' +
                '--config-args "{{bs: {bs}, lr: {lr}, end_lr: {end_lr}, att: 0}}" ' +
                '--output __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl ' +
                '--step {step} --section val --beam-size 1').format(
                    step=step,
                    bs=bs,
                    lr=lr,
                    end_lr=end_lr,
                    ))

            eval_command = ((
                'python eval.py --config configs/spider-20190205/nl2code-0428-stability.jsonnet ' +
                '--logdir logdirs/20190428-stability ' +
                '--config-args "{{bs: {bs}, lr: {lr}, end_lr: {end_lr}, att: 0}}" ' +
                '--inferred __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl ' +
                '--output __LOGDIR__/eval-val-step{step:05d}-bs1.jsonl ' +
                '--section val').format(
                    step=step,
                    bs=bs,
                    lr=lr,
                    end_lr=end_lr,
                    ))

            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
