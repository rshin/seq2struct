import argparse
import collections
import datetime
import json
import os

import _jsonnet
import attr
import asdl
import torch

from seq2struct import ast_util
from seq2struct import datasets
from seq2struct import models
from seq2struct import optimizers

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
    eval_on_train = attr.ib(default=True)
    eval_on_val = attr.ib(default=True)


class Logger:
    def __init__(self, log_path=None):
        self.log_file = None
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'a+')

    def log(self, msg):
        formatted = '[{}] {}'.format(
            datetime.datetime.now().replace(microsecond=0).isoformat(),
            msg)
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            self.log_file.flush()


def eval_model(logger, model, last_step, eval_data_loader, eval_section, num_eval_items=None):
    stats = collections.defaultdict(float)
    model.eval()
    with torch.no_grad():
      for eval_batch in eval_data_loader:
          batch_res = model.eval_on_batch(eval_batch)
          for k, v in batch_res.items():
              stats[k] += v
          if num_eval_items and stats['total'] > num_eval_items:
              break
    model.train()

    # Divide each stat by 'total'
    for k in stats:
        if k != 'total':
            stats[k] /= stats['total']
    del stats['total']

    logger.log("Step {} stats, {}: {}".format(
        last_step, eval_section, ", ".join(
        "{} = {}".format(k, v) for k, v in stats.items())))


def yield_batches_from_epochs(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])
    train_config = registry.instantiate(TrainConfig, config['train'])

    logger = Logger(os.path.join(args.logdir, 'log.txt'))

    # 0. Construct preprocessors
    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'],
        unused_keys=('name',))
    model_preproc.load()

    # 1. Construct model
    model = registry.construct('model', config['model'],
            unused_keys=('encoder_preproc', 'decoder_preproc'), preproc=model_preproc, device=device)
    model.to(device)

    optimizer = registry.construct('optimizer', config['optimizer'], params=model.parameters())
    lr_scheduler = registry.construct(
            'lr_scheduler', 
            config.get('lr_scheduler', {'name': 'noop'}),
            optimizer=optimizer)

    # 2. Restore its parameters
    saver = saver_mod.Saver(
        model, optimizer, keep_every_n=train_config.keep_every_n)
    last_step = saver.restore(args.logdir)

    # 3. Get training data somewhere
    train_data = model_preproc.dataset('train')
    train_data_loader = yield_batches_from_epochs(
        torch.utils.data.DataLoader(
            train_data,
            batch_size=train_config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: x))
    train_eval_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=train_config.eval_batch_size,
            collate_fn=lambda x: x)

    val_data = model_preproc.dataset('val')
    val_data_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=train_config.eval_batch_size,
            collate_fn=lambda x: x)

    # 4. Start training loop
    for batch in train_data_loader:
        # Quit if too long
        if last_step >= train_config.max_steps:
            break

        # Evaluate model
        if last_step % train_config.eval_every_n == 0:
            if train_config.eval_on_train:
                eval_model(logger, model, last_step, train_eval_data_loader, 'train', num_eval_items=train_config.num_eval_items)
            if train_config.eval_on_val:
                eval_model(logger, model, last_step, val_data_loader, 'val', num_eval_items=train_config.num_eval_items)
        
        # Compute and apply gradient
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        lr_scheduler.update_lr(last_step)
        optimizer.step()

        # Report metrics
        if last_step % train_config.report_every_n == 0:
            logger.log('Step {}: loss={:.4f}'.format(last_step, loss.item()))

        last_step += 1
        # Run saver
        if last_step % train_config.save_every_n == 0:
            saver.save(args.logdir, last_step)


if __name__ == '__main__':
    main()
