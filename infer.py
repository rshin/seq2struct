import argparse
import ast
import itertools
import json

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


def main():
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
    parser.add_argument('--mode', default='infer')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    # 0. Construct preprocessors
    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'])
    model_preproc.load()

    # 1. Construct model
    model = registry.construct('model', config['model'], preproc=model_preproc, device=device)
    model.to(device)
    model.eval()

    optimizer = registry.construct('optimizer', config['optimizer'], params=model.parameters())

    # 2. Restore its parameters
    saver = saver_mod.Saver(model, optimizer)
    last_step = saver.restore(args.logdir, step=args.step)
    if not last_step:
        raise Exception('Attempting to infer on untrained model')

    # 3. Get training data somewhere
    output = open(args.output, 'w')
    data = registry.construct('dataset', config['data'][args.section])
    if args.limit:
        sliced_data = itertools.islice(data, args.limit)
    else:
        sliced_data = data

    with torch.no_grad():
        if args.mode == 'infer':
            data = registry.construct('dataset', config['data'][args.section])
            if args.limit:
                sliced_data = itertools.islice(data, args.limit)
            else:
                sliced_data = data
            infer(model, args.beam_size, args.output_history, sliced_data, output)
        elif args.mode == 'debug':
            data = model_preproc.dataset(args.section)
            if args.limit:
                sliced_data = itertools.islice(data, args.limit)
            else:
                sliced_data = data
            debug(model, sliced_data, output)


def infer(model, beam_size, output_history, sliced_data, output):
    for i, item in enumerate(tqdm.tqdm(sliced_data)):
        beams = beam_search.beam_search(
                model, item, beam_size=beam_size, max_steps=1000)

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


if __name__ == '__main__':
    main()
