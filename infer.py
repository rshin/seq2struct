import argparse
import ast
import collections
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

    optimizer = registry.construct('optimizer', config['optimizer'], params=model.parameters())

    # 2. Restore its parameters
    saver = saver_mod.Saver(model, optimizer)
    last_step = saver.restore(args.logdir)

    # 3. Get training data somewhere
    train_data = model_preproc.dataset('train')

    # TODO don't assume EncDecModel
    enc_input, dec_output = train_data[0]
    enc_state = model.encoder(enc_input)

    traversal, choices = model.decoder.begin_inference(enc_state)
    for i in tqdm.tqdm(itertools.count()):
        #choices.sort(key=lambda c: c[1], reverse=True)
        #print('Step {}'.format(i))
        #print('- Top choices: {}'.format(choices[:3]))
        #print('- Current tree: {}'.format(traversal.actions))
        best_choice, best_score = max(choices, key=lambda c: c[1])
        choices = traversal.step(best_choice)
        if choices is None:
            break

    # TODO don't assume Hearthstone
    tree = traversal.finalize()
    ast_tree = ast_util.to_native_ast(tree)
    inferred_code = astor.to_source(ast_tree)
    canonicalized_gold_code = astor.to_source(ast.parse(dec_output.orig_code))

    import IPython
    IPython.embed()


if __name__ == '__main__':
    main()
