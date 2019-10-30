import json
import os

import tqdm
import _jsonnet

from seq2struct import datasets
from seq2struct.utils import registry

def compute_metrics(config_path, config_args, section, inferred_path,logdir=None, evaluate_beams_individually=False):
    if config_args:
        config = json.loads(_jsonnet.evaluate_file(config_path, tla_codes={'args': config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(config_path))

    if 'model_name' in config and logdir:
        logdir = os.path.join(logdir, config['model_name'])
    if logdir:
        inferred_path = inferred_path.replace('__LOGDIR__', logdir)

    inferred = open(inferred_path)
    data = registry.construct('dataset', config['data'][section])

    inferred_lines = list(inferred)
    if len(inferred_lines) < len(data):
        raise Exception('Not enough inferred: {} vs {}'.format(len(inferred_lines),
          len(data)))

    if evaluate_beams_individually:
        return logdir, evaluate_all_beams(data, inferred_lines)
    else:
        return logdir, evaluate_default(data, inferred_lines)

def load_from_lines(inferred_lines):
    for line in inferred_lines:
        infer_results = json.loads(line)
        if infer_results.get('beams', ()):
            inferred_code = infer_results['beams'][0]['inferred_code']
        else:
            inferred_code = None
        yield inferred_code, infer_results

def evaluate_default(data, inferred_lines):
    metrics = data.Metrics(data)
    for inferred_code, infer_results in load_from_lines(inferred_lines):
        if 'index' in infer_results:
            metrics.add(data[infer_results['index']], inferred_code)
        else:
            metrics.add(None, inferred_code, obsolete_gold_code=infer_results['gold_code'])

    return metrics.finalize()


def evaluate_all_beams(data, inferred_lines):
    metrics = data.Metrics(data)
    results = []
    for _, infer_results in load_from_lines(inferred_lines):
        for_beam = metrics.evaluate_all(
            infer_results['index'],
            data[infer_results['index']],
            [beam['inferred_code'] for beam in infer_results.get('beams', ())])
        results.append(for_beam)
    return results
