import argparse
import ast
import itertools
import json
import os
import sys

import _jsonnet
import asdl
import astor
import torch
import tqdm

from seq2struct import beam_search
from seq2struct import datasets
from seq2struct import models
from seq2struct import optimizers
from seq2struct.utils import registry
from seq2struct.utils import saver as saver_mod


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--config-args')

parser.add_argument('--step', type=int)
parser.add_argument('--section', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--beam-size', required=True, type=int)
parser.add_argument('--output-history', action='store_true')
parser.add_argument('--limit', type=int)
parser.add_argument('--mode', default='infer', choices=['infer', 'debug', 'visualize_attention'])
parser.add_argument('--res1', default='outputs/glove-sup-att-1h-0/outputs.json')
parser.add_argument('--res2', default='outputs/glove-sup-att-1h-1/outputs.json')
parser.add_argument('--res3', default='outputs/glove-sup-att-1h-2/outputs.json')
args = parser.parse_args()

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(1)
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    output_path = args.output.replace('__LOGDIR__', args.logdir)
    if os.path.exists(output_path):
        print('Output file {} already exists'.format(output_path))
        sys.exit(1)

    # 0. Construct preprocessors
    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'])
    model_preproc.load()

    # 1. Construct model
    model = registry.construct('model', config['model'], preproc=model_preproc, device=device)
    model.to(device)
    model.eval()
    model.visualize_flag = False

    optimizer = registry.construct('optimizer', config['optimizer'], params=model.parameters())

    # 2. Restore its parameters
    saver = saver_mod.Saver(model, optimizer)
    last_step = saver.restore(args.logdir, step=args.step, map_location=device)
    if not last_step:
        raise Exception('Attempting to infer on untrained model')

    # 3. Get training data somewhere
    output = open(output_path, 'w')
    data = registry.construct('dataset', config['data'][args.section])
    if args.limit:
        sliced_data = itertools.islice(data, args.limit)
    else:
        sliced_data = data

    with torch.no_grad():
        if args.mode == 'infer':
            orig_data = registry.construct('dataset', config['data'][args.section])
            preproc_data = model_preproc.dataset(args.section)
            if args.limit:
                sliced_orig_data = itertools.islice(data, args.limit)
                sliced_preproc_data = itertools.islice(data, args.limit)
            else:
                sliced_orig_data = orig_data
                sliced_preproc_data = preproc_data
            assert len(orig_data) == len(preproc_data)
            infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data, output)
        elif args.mode == 'debug':
            data = model_preproc.dataset(args.section)
            if args.limit:
                sliced_data = itertools.islice(data, args.limit)
            else:
                sliced_data = data
            debug(model, sliced_data, output)
        elif args.mode == 'visualize_attention':
            model.visualize_flag = True
            model.decoder.visualize_flag = True
            data = registry.construct('dataset', config['data'][args.section])
            if args.limit:
                sliced_data = itertools.islice(data, args.limit)
            else:
                sliced_data = data
            visualize_attention(model, args.beam_size, args.output_history, sliced_data, output)


def infer(model, beam_size, output_history, sliced_orig_data, sliced_preproc_data, output):
    for i, (orig_item, preproc_item) in enumerate(
            tqdm.tqdm(zip(sliced_orig_data, sliced_preproc_data),
                      total=len(sliced_orig_data))):
        beams = beam_search.beam_search(
                model, orig_item, preproc_item, beam_size=beam_size, max_steps=1000)

        decoded = []
        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize()

            decoded.append({
                'model_output': model_output,
                'inferred_code': inferred_code,
                'score': beam.score,
                **({
                    'choice_history': beam.choice_history,
                    'score_history': beam.score_history,
                } if output_history else {})})

        output.write(
            json.dumps({
                'index': i,
                'beams': decoded,
            }) + '\n')
        output.flush()


def debug(model, sliced_data, output):
    for i, item in enumerate(tqdm.tqdm(sliced_data)):
        (_, history), = model.compute_loss([item], debug=True)
        output.write(
                json.dumps({
                    'index': i,
                    'history': history,
                }) + '\n')
        output.flush()

def visualize_attention(model, beam_size, output_history, sliced_data, output):
    res1 = json.load(open(args.res1, 'r'))
    res1 = res1['per_item']
    res2 = json.load(open(args.res2, 'r'))
    res2 = res2['per_item']
    res3 = json.load(open(args.res3, 'r'))
    res3 = res3['per_item']
    interest_cnt = 0
    cnt = 0
    for i, item in enumerate(tqdm.tqdm(sliced_data)):
        
        if res1[i]['hardness'] != 'extra':
            continue
        
        cnt += 1
        if (res1[i]['exact'] == 0) and (res2[i]['exact'] == 0) and (res3[i]['exact'] == 0):
            continue
        interest_cnt += 1
        '''
        print('sample index: ')
        print(i)
        beams = beam_search.beam_search(
            model, item, beam_size=beam_size, max_steps=1000, visualize_flag=True)
        entry = item.orig
        print('ground truth SQL:')
        print(entry['query_toks'])
        print('prediction:')
        print(res2[i])
        decoded = []
        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize()

            decoded.append({
                'model_output': model_output,
                'inferred_code': inferred_code,
                'score': beam.score,
                **({
                    'choice_history': beam.choice_history,
                    'score_history': beam.score_history,
                } if output_history else {})})

        output.write(
            json.dumps({
                'index': i,
                'beams': decoded,
            }) + '\n')
        output.flush()
        '''
    print(interest_cnt * 1.0 / cnt)

if __name__ == '__main__':
    main()
