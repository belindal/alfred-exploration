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
python models/train/train_transformer.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 4 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --save_path temp/transformer.pth --fast_epoch --save_path models/pretrained/transformer.pth

python3 models/train/train_transformer.py --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits /home/sahit/alfred-exploration/data/splits/oct21.json --gpu --batch 4 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --save_path temp/transformer.pth --fast_epoch --save_path /home/sahit/alfred-exploration/models/pretrained/transformer.pth
```
Add `--preprocess` the first time you run. In subsequent runs, you can remove this flag.

Add `--fast_epoch` for debugging.

This will by default train on a 1000-file subset of the training data (out of 21023 files total). Can increase this number by modifying this LOC: https://github.com/belindal/alfred-exploration/blob/main/models/train/train_transformer.py#L426

The main files that I added for the Transformer training are `models/train/train_transformer.py` and `models/model/t5.py`.

## Evaluating the Transformer
```bash
python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/pretrained/transformer_ep0_step24000_new.pth --eval_split valid_seen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1
```

- If you want to follow expert demonstrations until the last subgoal, add the flag `--force_last_subgoal`
- The list of instructions we use is found in `eval_task.py:65` (e.g from `traj['turk_annotations']['anns']`), not sure if these instructions are different than the ones used in training which seem to be `ex['ann']['instr']`
- When this starts working for real, increase num_threads from 1 to e.g 3


## Additional required setup:
- install allennlp
- install pytorch-lightning

- Add MaskRCNN weights:
```bash
mkdir -p storage/models/vision/moca_maskrcnn;
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt -O storage/models/vision/moca_maskrcnn/weight_maskrcnn.pt; 
```
