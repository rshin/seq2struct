import argparse
import ast
import collections
import copy
import datetime
import itertools
import json
import os

import _jsonnet
import attr
import asdl
import astor
import torch
import tqdm

from seq2struct import ast_util
from seq2struct import beam_search
from seq2struct.utils import registry
from seq2struct.utils import saver as saver_mod
from seq2struct.utils import vocab


@attr.s
class TrainConfig:
    eval_every_n = attr.ib(default=100)
    report_every_n = attr.ib(default=100)
    save_every_n = attr.ib(default=100)
    keep_every_n = attr.ib(default=1000)

    batch_size = attr.ib(default=32)
    eval_batch_size = attr.ib(default=32)
    max_steps = attr.ib(default=100000)
    num_eval_items = attr.ib(default=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)

    parser.add_argument('--section', default='train')
    parser.add_argument('--output', required=True)
    parser.add_argument('--beam-size', required=True, type=int)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
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
    last_step = saver.restore(args.logdir)

    # 3. Get training data somewhere
    output = open(args.output, 'w')

    data = model_preproc.dataset(args.section)
    for item in tqdm.tqdm(itertools.islice(data, args.limit)):
        beams = beam_search.beam_search(
                model, item, beam_size=args.beam_size, max_steps=1000)

        decoded = []
        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize()

            decoded.append({
                # TODO remove deepcopy
                'model_output': copy.deepcopy(model_output),
                'inferred_code': inferred_code,
                'score': beam.score,
                'choice_history': beam.choice_history,
                'score_history': beam.score_history,
            })

        # TODO don't assume EncDecModel, Hearthstone
        enc_input, dec_output = item
        canonicalized_gold_code = astor.to_source(
            ast.parse(dec_output.orig_code))
        output.write(
            json.dumps({
                'beams': decoded,
                'gold_code': canonicalized_gold_code
            }) + '\n')
        output.flush()


if __name__ == '__main__':
    main()
