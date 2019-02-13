import json

import _jsonnet

from seq2struct import datasets
from seq2struct.utils import registry

def compute_metrics(config_path, section, inferred_path):
    config = json.loads(_jsonnet.evaluate_file(config_path))

    inferred = open(inferred_path)
    data = registry.construct('dataset', config['data'][section])
    metrics = data.Metrics(data)

    for line in inferred:
        infer_results = json.loads(line)
        if infer_results['beams']:
            inferred_code = infer_results['beams'][0]['inferred_code']
        else:
            inferred_code = None
        if 'index' in infer_results:
            metrics.add(data[infer_results['index']], inferred_code)
        else:
            metrics.add(None, inferred_code, obsolete_gold_code=infer_results['gold_code'])

    return metrics.finalize()


