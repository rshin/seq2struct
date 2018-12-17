import argparse
import json
import os

import _jsonnet
import asdl
import torch

from seq2struct import ast_util
from seq2struct.utils import registry
from seq2struct.utils import saver as saver_mod
from seq2struct.utils import vocab


def yield_batches_from_epochs(loader):
    while True:
        for batch in loader:
            yield batch


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
    train_config = config['train']

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
    saver = saver_mod.Saver(
        model, optimizer, keep_every_n=train_config['keep_every_n'])
    last_step = saver.restore(args.logdir)

    # 3. Get training data somewhere
    train_data_loader = yield_batches_from_epochs(
        torch.utils.data.DataLoader(
            model_preproc.dataset('train'),
            batch_size=train_config['batch_size'],
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: x))
    val_data = torch.utils.data.DataLoader(
            model_preproc.dataset('val'),
            batch_size=train_config['batch_size'],
            collate_fn=lambda x: x)

    # 4. Start training loop
    for batch in train_data_loader:
        # Quit if too long
        if last_step >= train_config['max_steps']:
            break

        # Evaluate model
        if last_step % train_config['eval_every_n'] == 0:
            pass

        # Compute and apply gradient
        # TODO: update learning rate
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()

        # Report metrics
        if last_step % train_config['report_every_n'] == 0:
            print('Step {}: loss={:.4f}'.format(last_step, loss.item()))

        last_step += 1
        # Run saver
        if last_step % train_config['save_every_n'] == 0:
            saver.save(args.logdir, last_step)


if __name__ == '__main__':
    main()
