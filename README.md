# Setup
This repository requires Python 3.5 or greater.

Example instructions to set up:
```
virtualenv -p python3 /path/to/venv
git clone https://github.com/rshin/seq2struct
cd seq2struct
pip install -e .
```

# Dependencies
Required Python modules are specified in `requirements.txt`. This project currently uses PyTorch 0.4.

To train models for Spider, you also need the JVM to run Stanford CoreNLP (currently used for tokenization for GloVe embeddings).

# [Spider dataset](https://yale-lily.github.io/spider)
To obtain the results in https://arxiv.org/abs/1906.11790, first download the Spider dataset and preprocess it:
- Download “Spider Dataset” from https://yale-lily.github.io/spider. You can use this bash function:
```
function gdrive_download () {
  COOKIES=$(mktemp)
  CONFIRM=$(wget --quiet --save-cookies ${COOKIES} --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --content-disposition --load-cookies ${COOKIES} "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1"
  rm -rf ${COOKIES}
}
```
like `gdrive_download 11icoH_EA-NYb0OrPTdehRWm_d7-DIzWX`
- Unzip it somewhere
- Run `bash data/spider-20190206/generate.sh /path/to/unzipped/spider`
- Run `python preprocess.py --config configs/spider-20190205/arxiv-1906.11790v1.jsonnet`

Install Stanford CoreNLP:
- Download http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
- Unzip it to `third_party/stanford-corenlp-full-2018-10-05`

To train the model:
```
python train.py --config configs/spider-20190205/arxiv-1906.11790v1.jsonnet --logdir ../logs/arxiv-1906.11790v1
```
This should create a directory `../logs/arxiv-1906.11790v1`.

To perform inference:
```
python infer.py --config configs/spider-20190205/arxiv-1906.11790v1.jsonnet --logdir ../logs/arxiv-1906.11790v1 --step <STEP NUMBER> --section val --beam-size 1 --output <PATH FOR INFERENCE OUTPUT>
```

To perform evaluation:
```
python eval.py --config configs/spider-20190205/arxiv-1906.11790v1.jsonnet --inferred <PATH FOR INFERENCE OUTPUT> --output <PATH FOR EVAL OUTPUT> --section val
```

To look at evaluation results:
```
>>> import json
>>> d = json.load(open('<PATH FOR EVAL OUTPUT>')) 
>>> print(d['total_scores']['all']['exact']) # should be ~0.42
```