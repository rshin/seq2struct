import argparse
import ast
import itertools
import functools
import json
import os
import sys

import _jsonnet
import asdl
import astor
import torch
import tqdm
import multiprocessing

from seq2struct import beam_search
from seq2struct import datasets
from seq2struct import models
from seq2struct import optimizers
from seq2struct.utils import registry
from seq2struct.utils import saver as saver_mod
from seq2struct.utils import parallelizer

class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)

        # 0. Construct preprocessors
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        self.model_preproc.load()

    def load_model(self, logdir, step):
        '''Load a model (identified by the config used for construction) and return it'''
        # 1. Construct model
        model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device)
        model.to(self.device)
        model.eval()
        model.visualize_flag = False

        optimizer = registry.construct('optimizer', self.config['optimizer'], params=model.parameters())

        # 2. Restore its parameters
        saver = saver_mod.Saver(model, optimizer)
        last_step = saver.restore(logdir, step=step, map_location=self.device)
        if not last_step:
            raise Exception('Attempting to infer on untrained model')
        return model

    def infer(self, model, output_path, args):
        # 3. Get training data somewhere
        output = open(output_path, 'w')
        data = registry.construct('dataset', self.config['data'][args.section])
        if args.limit:
            sliced_data = itertools.islice(data, args.limit)
        else:
            sliced_data = data

        with torch.no_grad():
            if args.mode == 'infer':
                orig_data = registry.construct('dataset', self.config['data'][args.section])
                preproc_data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_orig_data = itertools.islice(data, args.limit)
                    sliced_preproc_data = itertools.islice(data, args.limit)
                else:
                    sliced_orig_data = orig_data
                    sliced_preproc_data = preproc_data
                assert len(orig_data) == len(preproc_data)
                self._inner_infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data, output, args.nproc)
            elif args.mode == 'debug':
                data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._debug(model, sliced_data, output)
            elif args.mode == 'visualize_attention':
                model.visualize_flag = True
                model.decoder.visualize_flag = True
                data = registry.construct('dataset', self.config['data'][args.section])
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._visualize_attention(model, args.beam_size, args.output_history, sliced_data, args.res1, args.res2, args.res3, output)

    def _inner_infer(self, model, beam_size, output_history, sliced_orig_data, sliced_preproc_data, output, nproc):
        list_items = [(idx, oi, pi) for idx, (oi, pi) in enumerate(zip(sliced_orig_data, sliced_preproc_data))]
        total = len(sliced_orig_data)

        cp = parallelizer.CPUParallelizer(nproc)
        params = [
            (beam_size, output_history, indices, orig_items, preproc_items)
                for indices, orig_items, preproc_items in list_items
        ]
        write_all(output, cp.parallel_map([(functools.partial(self._infer_single, model), params)]))

    def _infer_single(self, model, param):

        beam_size, output_history, index, orig_item, preproc_item = param

        try:
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
            result = {
                'index': index,
                'beams': decoded,
            }
        except Exception as e:
            result = {
                'index': index,
                'error': str(e),
            }
        return json.dumps(result) + '\n'

    def _debug(self, model, sliced_data, output):
        for i, item in enumerate(tqdm.tqdm(sliced_data)):
            (_, history), = model.compute_loss([item], debug=True)
            output.write(
                    json.dumps({
                        'index': i,
                        'history': history,
                    }) + '\n')
            output.flush()

    def _visualize_attention(self, model, beam_size, output_history, sliced_data, res1file, res2file, res3file, output):
        res1 = json.load(open(res1file, 'r'))
        res1 = res1['per_item']
        res2 = json.load(open(res2file, 'r'))
        res2 = res2['per_item']
        res3 = json.load(open(res3file, 'r'))
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

def write_all(output, genexp):
    for item in genexp:
        output.write(item)
        output.flush()

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
    parser.add_argument('--mode', default='infer', choices=['infer', 'debug', 'visualize_attention'])
    parser.add_argument('--res1', default='outputs/glove-sup-att-1h-0/outputs.json')
    parser.add_argument('--res2', default='outputs/glove-sup-att-1h-1/outputs.json')
    parser.add_argument('--res3', default='outputs/glove-sup-att-1h-2/outputs.json')
    parser.add_argument('--nproc', type=int, default=1)
    args = parser.parse_args()

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

    inferer = Inferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)


if __name__ == '__main__':
    main()
