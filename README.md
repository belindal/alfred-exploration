# ALFRED

For basic ALFRED setup, following instructions under Quickstart in ALFRED repo's README (https://github.com/askforalfred/alfred), copied here:

## ALFRED Quickstart

Clone repo:
```bash
$ git clone https://github.com/askforalfred/alfred.git alfred
$ export ALFRED_ROOT=$(pwd)/alfred
```

Install requirements:
```bash
$ virtualenv -p $(which python3) --system-site-packages alfred_env # or whichever package manager you prefer
$ source alfred_env/bin/activate

$ cd $ALFRED_ROOT
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

Download Trajectory JSONs and Resnet feats (~17GB):
```bash
$ cd $ALFRED_ROOT/data
$ sh download_data.sh json_feat
```


## Training the transformer
```bash
python models/train/train_transformer.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
```
Add `--preprocess` the first time you run. In subsequent runs, you can remove this flag.

This will by default train on a 1000-file subset of the training data (out of 21023 files total). Can increase this number by modifying this LOC: https://github.com/belindal/alfred-exploration/blob/main/models/train/train_transformer.py#L426

The main files that I added for the Transformer training are `models/train/train_transformer.py` and `models/model/t5.py`.

## Evaluating the Transformer
```bash
python3 models/eval/eval_seq2seq.py --model_path models/pretrained/transformer.pth --eval_split valid_seen --data data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1``

- When this starts working for real, increase num_threads from 1 to e.g 3
- Also when this starts working we can delete the call to sys.exit() in eval_task.py

Rough summary of changes made in this branch
- move the super constructor call in the ALFREDDataLoader (models/train/train_transformer.py) to the end of the constructor (without this I was getting complaints about self.dataset being changed)
- added saving/loading model functionality in train_transformer.py and models/model/t5.py respectively
- modify test_generate in t5.py so that image_sequence_padded was the right shape
- add a featurize function to t5.py that takes information about the episode and connverts it into a form that can be part of the input to test_generate
- modify models/eval/eval_task.py to load the GoalConditionedTransformer and pass output from the environment/task description into the model and then decode the output (the main changes happened here)
