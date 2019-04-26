To set up:
```
virtualenv -p python3 /path/to/venv # Requires Python 3
git clone ...
cd seq2struct
pip install -e .
```

To set up the data:
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
- Run `python preprocess.py --config configs/spider-20190205/nl2code-0220.jsonnet --config-args "{output_from: false, qenc: 'eb', ctenc: 'ebs', upd_steps: 4, max_steps: 40000, batch_size: 10}"`

To train the model:
`python train.py --config configs/spider-20190205/nl2code-0220.jsonnet --config-args "{output_from: false, qenc: 'eb', ctenc: 'ebs', upd_steps: 4, max_steps: 40000, batch_size: 10}" --logdir ../logs`
This should create a directory `../logs/output_from=false,qenc=eb,ctenc=ebs,upd_steps=4,max_steps=40000,batch_size=10/`.

To perform inference:
`python infer.py --config configs/spider-20190205/nl2code-0220.jsonnet --config-args "{output_from: false, qenc: 'eb', ctenc: 'ebs', upd_steps: 4, max_steps: 40000, batch_size: 10}" --logdir ../logs/ --step <STEP NUMBER> --section val --beam-size 1 --output <PATH FOR INFERENCE OUTPUT>`

To perform evaluation:
`python eval.py --config configs/spider-20190205/nl2code-0220.jsonnet --config-args "{output_from: false, qenc: 'eb', ctenc: 'ebs', upd_steps: 4, max_steps: 40000, batch_size: 10}" --inferred <PATH FOR INFERENCE OUTPUT> --output <PATH FOR EVAL OUTPUT> --section val`

To look at evaluation results:
```
>>> import json
>>> d = json.load(open('<PATH FOR EVAL OUTPUT>')) 
>>> print(d['total_scores']['all']['exact']) # should be in 0.20-0.27
```