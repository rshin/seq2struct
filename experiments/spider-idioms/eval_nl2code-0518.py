import itertools
import json
import os
from pathlib import Path

import _jsonnet

PT_RESULTS = Path.home() / "spider/pt-results/2019-05-21/"


def main():
    for job in PT_RESULTS.iterdir():
        if not job.is_dir(): continue
        
        for st, nt, att in itertools.product(
                ('cov-xent', 'cov-examples'),
                (10, 20, 40, 80),
                (0,),
        ):
            logdir = job / f"logdirs/20190519/filt-none_st-{st}_nt-{nt}"
            if not logdir.exists(): continue
            
            steps = list(range(33000, 0, -100))
            args = '{{st: \'{st}\', nt: {nt}, att: {att}}}'.format(
                st=st,
                nt=nt,
                att=att)
    
            # config = json.loads(
            #     _jsonnet.evaluate_file(
            #         'configs/spider-idioms/nl2code-0518.jsonnet', tla_codes={'args': args}))
            # 
            for step in steps:
                if not os.path.exists(os.path.join(
                        logdir,
                        'model_checkpoint-{:08d}'.format(step))):
                    continue
    
                if os.path.exists(os.path.join(
                        logdir,
                        'eval-val-step{:05d}-bs1.jsonl'.format(step))):
                    continue
    
                infer_command = ((
                    'python infer.py --config configs/spider-idioms/nl2code-0518.jsonnet '
                    '--logdir logdirs/spider-idioms/nl2code-0518 '
                    '--config-args "{args}" '
                    '--output __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl '
                    '--step {step} --section val --beam-size 1').format(
                    args=args,
                    step=step,
                ))
    
                eval_command = ((
                    'python eval.py --config configs/spider-idioms/nl2code-0518.jsonnet '
                    '--logdir logdirs/spider-idioms/nl2code-0518 '
                    '--config-args "{args}" '
                    '--inferred __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl '
                    '--output __LOGDIR__/eval-val-step{step:05d}-bs1.jsonl '
                    '--section val').format(
                    args=args,
                    step=step,
                ))
                print('{} && {}'.format(infer_command, eval_command))
                # print(eval_command)
                # print(infer_command)


if __name__ == '__main__':
    main()
