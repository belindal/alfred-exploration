import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import os
import torch
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to
from models.nn.vnn import ResnetVisualEncoder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import trange
import collections
import numpy as np
from gen.utils.image_util import decompress_mask
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW
from models.model.t5 import GoalConditionedTransformer, unCamelSnakeCase, API_ACTIONS_NATURALIZED, API_ACTIONS_SN, CLASSES_NATURALIZED
from tqdm import tqdm
import itertools as it
import random
import regex as re
random.seed(0)


# python models/train/train_transformer.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
class ALFREDDataloader(DataLoader):
    """
    *_seq_past: previous states/actions before the curr state (i.e. [a,b,c])
    *_curr; curr (single) state/action to output (i.e. d)
    *_seq_w_curr: sequence of states/actions including the curr one (i.e. [a,b,c,d])
    """
    def __init__(self, args, vocab, dataset, split_type, batch_size, curr_subgoal_only: bool=False):
        self.vocab = vocab
        self.args = args
        # params
        self.max_subgoals = 25
        self.feat_pt = 'feat_conv.pt'
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.pad = self.tokenizer.pad_token_id
        self.image_input_dim = 512
        self.vis_encoder = ResnetVisualEncoder(dframe=self.image_input_dim)
        for param in self.vis_encoder.parameters():
            param.requires_grad = False
        self.action_mask_dim = None
        self.curr_subgoal_only = curr_subgoal_only

        if "train" in split_type or "valid" in split_type:
            # self.dataset = self.featurize(self.dataset)
            new_dataset = []
            for task in tqdm(dataset):
                ex = self.load_task_json(task, args.data, args.pp_folder)
                goal = [g.rstrip() for g in ex['ann']['goal']]

                # load Resnet features from disk
                root = self.get_task_root(ex)
                im = torch.load(os.path.join(root, self.feat_pt))
                num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action
                num_feat_frames = im.shape[0]
                # Modeling Quickstart (without filler frames)
                if num_low_actions == num_feat_frames:
                    all_frames = im
                # Full Dataset (contains filler frames)
                else:
                    keep = [None] * num_low_actions
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep[-1] = im[-1]  # stop frame
                    all_frames = torch.stack(keep, dim=0)

                # convert resnet features to input features
                all_frames = self.vis_encoder(all_frames)
                assert all_frames.size(0) == num_low_actions

                assert len(ex['ann']['instr']) == len(ex['num']['action_low'])
                prev_subgoals = []
                action_sequence = []
                action_args_sequence = []
                action_mask_sequence = []
                all_subgoals = [[sg.rstrip() for sg in subgoal] for subgoal in ex['ann']['instr']]
                all_subgoals_flattened = list(it.chain(*all_subgoals))
                frame_idx = 0
                api_actions_by_subgoal = self.get_api_actions_by_subgoals(ex, all_subgoals)
                for subgoal_idx, subgoal in enumerate(all_subgoals):
                    curr_subgoal_action_idxs = []

                    # in terms of the action vocabulary
                    # subgoal_actions = self.vocab['action_high'].index2word(ex['num']['action_high'][subgoal_idx]['action'])
                    for action_idx, action in enumerate(ex['num']['action_low'][subgoal_idx]):
                        action_low = self.vocab['action_low'].index2word(action['action'])
                        if "MoveAhead" in action_low:
                            # Downsample moveahead
                            if random.random() > 0.25: continue
                        action_nl = action_low
                        if action_low == '<<stop>>':
                            # special case where api_actions_by_subgoal[subgoal_idx] will be empty
                            action_args_sequence.append({})
                        else:
                            if (action_idx >= len(api_actions_by_subgoal[subgoal_idx])) or (subgoal_idx >= len(api_actions_by_subgoal)):
                                # For whatever reason there is one misaligned example
                                continue
                            api_action = api_actions_by_subgoal[subgoal_idx][action_idx]
                            assert (
                                api_action["action"] == action_low
                            ), "Action must match to action_low otherwise our action_args sequence is all screwy"
                            action_args = self.process_action_args(api_action)
                            action_args_sequence.append(action_args)
                            if 'object' in action_args:
                                action_nl += ": " + action_args['object'].strip()
                            if 'receptacle' in action_args:
                                action_nl += " in " + action_args['receptacle']
                        action_nl = unCamelSnakeCase(action_nl).replace('  ', ' ')+','
                        action_sequence.append(action_nl)
                        if action['mask'] is not None:
                            assert action['valid_interact'] == 1
                        else:
                            assert action['valid_interact'] == 0
                        curr_action_mask = self.decompress_mask(action['mask']) if action['mask'] is not None else None
                        if curr_action_mask is not None and self.action_mask_dim is None:
                            self.action_mask_dim = curr_action_mask.size()
                        action_mask_sequence.append(curr_action_mask)
                        curr_subgoal_action_idxs.append(frame_idx)
                        frame_idx += 1
                        new_ex = {
                            "goal": goal,
                            # make copies of lists
                            "all_subgoals": all_subgoals_flattened,
                            "prev_subgoals": [sg for sg in prev_subgoals],
                            "curr_subgoal": subgoal,
                            "curr_subgoal_action_idxs": [action_idx for action_idx in curr_subgoal_action_idxs],
                            "state_seq": all_frames[:frame_idx],
                            "action_seq": [a for a in action_sequence],
                            "action_mask_seq": [am for am in action_mask_sequence],
                            "action_args_seq": [arg for arg in action_args_sequence]
                        }
                        assert len(new_ex['state_seq']) == len(new_ex['action_seq']) == len(new_ex['action_mask_seq']) == len(new_ex['action_args_seq']) == new_ex['curr_subgoal_action_idxs'][-1] + 1
                        new_dataset.append(new_ex)

                    # add token for end of subgoal
                    if self.curr_subgoal_only and subgoal_idx != len(all_subgoals) - 1:
                        if len(curr_subgoal_action_idxs) == 0:
                            # skip this subgoal
                            continue
                        new_ex = {
                            "curr_subgoal": subgoal,
                            "curr_subgoal_action_idxs": [action_idx for action_idx in curr_subgoal_action_idxs] + [curr_subgoal_action_idxs[-1] + 1],
                            "state_seq": torch.cat([all_frames[:frame_idx], all_frames[frame_idx-1].unsqueeze(0)], dim=0),
                            "action_seq": [a for a in action_sequence] + ["[subgoal]"],
                            "action_mask_seq": [am for am in action_mask_sequence] + [action_mask_sequence[-1]],
                            "action_args_seq": [arg for arg in action_args_sequence] + [{}],
                        }
                        new_dataset.append(new_ex)

                    prev_subgoals += subgoal
        else:
            new_dataset = []
            for task in tqdm(dataset):
                new_dataset.append(self.load_task_json(task, args.data, args.pp_folder))
        random.shuffle(new_dataset)
        super().__init__(new_dataset, batch_size, collate_fn=self.collate_fn)

    def process_action_args(self, api_action):
        '''
        example api_action:
        {
            "action": "PutObject",
            "forceAction": True,
            "objectId": "ButterKnife|+00.29|+00.93|+00.99",
            "placeStationary": True,
            "receptacleObjectId": "Mug|-00.16|+00.93|+00.08",
        }
        '''
        args = {}
        if 'objectId' in api_action and api_action['objectId'].find('|') > 0:
            objId = api_action['objectId']
            args['object'] = objId[:objId.find('|')]
        if 'receptacleObjectId' in api_action and api_action['receptacleObjectId'].find('|') > 0:
            recpId = api_action['receptacleObjectId']
            args['receptacle'] = recpId[:recpId.find('|')]
        return args

    def get_api_actions_by_subgoals(self, ex, all_subgoals):
        api_actions_by_subgoal = {subgoal_idx: [] for subgoal_idx in range(len(all_subgoals))}
        for la in ex["plan"]["low_actions"]:
            api_action = la["api_action"]
            api_action['action'] = la["discrete_action"]['action']
            api_actions_by_subgoal[la["high_idx"]] = api_actions_by_subgoal[
                la["high_idx"]
            ] + [api_action]
        return api_actions_by_subgoal

    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        # mask = np.expand_dims(mask, axis=0)
        return torch.tensor(mask)

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def load_task_json(self, task, data_path, pp_folder):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(data_path, task['task'], '%s' % pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def pad_stack(self, data: list, pad_id: float):
        if len(data) == 0:
            return torch.tensor([]).to('cuda') if self.args.gpu else torch.tensor([])
        max_item_dims = list(max([item.size() for item in data]))
        padded_data = []
        for item in data:
            while len(item.size()) < len(max_item_dims): item = item.unsqueeze(0)
            pad_dims = []
            for dim_idx in range(len(max_item_dims)-1,-1,-1):
                pad_dims += [0, max_item_dims[dim_idx]-item.size(dim_idx)]
            padded_data.append(F.pad(input=item, pad=tuple(pad_dims), value=pad_id))
        return torch.stack(padded_data)

    def collate_fn(self, batch):
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = {
            f'{seq_type}_{seq_span}': [] for seq_type in ['state', 'action', 'action_mask'] for seq_span in ['seq_w_curr', 'seq_past', 'curr']
        }
        feat['input_goals'] = []
        for item in batch:
            # 'goal', 'curr_subgoal', 'prev_subgoals', 'all_subgoals', '{state|action|action_mask}_{history|next|seq}', 'curr_subgoal_action_idxs'
            assert len(item['action_seq']) == len(item['state_seq']) == len(item['action_mask_seq']) == item['curr_subgoal_action_idxs'][-1] + 1
            # expand `state_seq` and `action_mask_seq` for each token of `action_seq`
            expanded_seqs = {f'{seq_type}_{seq_span}': [] for seq_type in ['state', 'action_mask'] for seq_span in ['seq_w_curr', 'seq_past', 'curr']}
            if self.curr_subgoal_only:
                feat['input_goals'].append(''.join(item['curr_subgoal']).replace('<<goal>>', ' [goal]').replace('<<stop>>', ' [stop]').replace('  ', ' ').strip())
                action_idxs_for_goal = item['curr_subgoal_action_idxs']
                action_sequence = [item['action_seq'][idx] for idx in action_idxs_for_goal]
            else:
                feat['input_goals'].append(''.join(item['goal'] + item['all_subgoals']).replace('<<goal>>', ' [goal]').replace('<<stop>>', ' [stop]').replace('  ', ' ').strip())
                action_idxs_for_goal = list(range(len(item['action_seq'])))
                action_sequence = item['action_seq']

            for action_idx in action_idxs_for_goal:
                for _ in self.tokenizer.tokenize(item['action_seq'][action_idx]):
                    for seq_type in ['state', 'action_mask']:
                        if item[f"{seq_type}_seq"][action_idx] is None:
                            assert seq_type == 'action_mask'
                            item[f"{seq_type}_seq"][action_idx] = torch.full(self.action_mask_dim, -100)  # (1, 300, 300)
                        expanded_seqs[f'{seq_type}_seq_past' if action_idx < action_idxs_for_goal[-1] - 1 else f'{seq_type}_curr'].append(item[f"{seq_type}_seq"][action_idx])
                        expanded_seqs[f'{seq_type}_seq_w_curr'].append(item[f"{seq_type}_seq"][action_idx])
            for seq_key in expanded_seqs.keys():
                feat[seq_key].append(
                    torch.stack(expanded_seqs[seq_key]).to(device)
                    if len(expanded_seqs[seq_key]) > 0 else torch.tensor([]).to(device)
                )

            feat['action_seq_w_curr'].append(' '.join(action_sequence))
            feat['action_seq_past'].append(' '.join(action_sequence[:-1]))
            feat['action_curr'].append(action_sequence[-1])

            # (n_actions, 512)
            assert feat['state_seq_w_curr'][-1].size(0) == len(self.tokenizer.tokenize(feat['action_seq_w_curr'][-1]))
            # (n_actions, 300, 300)
            assert feat['action_mask_seq_w_curr'][-1].size(0) == feat['state_seq_w_curr'][-1].size(0)
        for key in feat:
            if type(feat[key][0]) == str:
                feat_key_lens = [len(item) for item in feat[key]]
                if max(feat_key_lens) == 0:
                    feat[key] = {'input_ids': torch.Tensor(len(batch),0).to(device).long(), 'attention_mask': torch.Tensor(len(batch),0).to(device).long()}
                else:
                    feat[key] = self.tokenizer(feat[key], return_tensors='pt', padding=True, add_special_tokens=(False if 'action' in key else True)).to(device)
            elif type(feat[key][0]) == torch.Tensor:
                feat[key] = self.pad_stack(feat[key], pad_id=-100 if key=='mask' else 0)
            else:
                assert False, f"features should be of type str or tensor, but got type {type(feat[key][0])} for feature {key}"
        return batch, feat

    def compute_metrics(self, preds, data):
        n_correct = 0.0
        n_total = 0
        output_dicts = []
        for idx, pred in enumerate(preds):
            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            pred = pred.split(',')[0].strip()  # segment out first generated action
            if pred.startswith("[subgoal]"):
                pred = "[subgoal]"
            gt = self.tokenizer.decode(data['action_curr']['input_ids'][idx], skip_special_tokens=True).strip(',')
            output_dicts.append({
                'pred': pred,
                'gt': gt,
            })
            n_correct += pred == gt
            n_total += 1
        return {'accuracy': n_correct / n_total, 'output_dicts': output_dicts}


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--train_size', help='amount of training data to use', default=1000, type=int)
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='splits/oct21.json')
    parser.add_argument('--save_path', help='path to save to', default='models/pretrained/transformer.pth')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_im')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')
    parser.add_argument('--use_templated_goals', help='use templated goals instead of human-annotated goal descriptions (only available for train set)', action='store_true')
    parser.add_argument('--resume', help='load a checkpoint')
    parser.add_argument('--curr_subgoal_only', help='store preprocessed data to json files', action='store_true')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=8, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=20, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=2500, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0., type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0., type=float)
    parser.add_argument('--label_smoothing', help='amount of label smoothing to use (0 by default)', default=0., type=float)

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)', default=0, type=int)

    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)

    # check if dataset has been preprocessed
    if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
        raise Exception("Dataset not processed; run with --preprocess")

    # make output dir
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    # preprocess and save
    if args.preprocess:
        print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % args.pp_folder)
        dataset = Dataset(args, None)
        dataset.preprocess_splits(splits)
        vocab = torch.load(os.path.join(args.dout, "%s.vocab" % args.pp_folder))
    else:
        # dataset = Dataset(args, None)
        vocab = torch.load(os.path.join(args.data, "%s.vocab" % args.pp_folder))

    dl_splits = {}
    for split_type in splits:
        # if split_type == "train":
        #     sep_actions = True
        # else:
        print(f"Processing {split_type} dataset")
        if args.fast_epoch:
            splits[split_type] = splits[split_type][:5]
        elif split_type == "train":
            splits[split_type] = random.sample(splits[split_type], args.train_size)
        else:
            splits[split_type] = random.sample(splits[split_type], 50)
        dl_splits[split_type] = ALFREDDataloader(args, vocab, splits[split_type], split_type, args.batch, curr_subgoal_only=args.curr_subgoal_only)

    # load model
    # model = T5ForConditionalGeneration.from_pretrained('t5-small').to('cuda')
    save_path = args.save_path
    if os.path.exists(save_path):
        model = GoalConditionedTransformer.load(args, save_path)
    else:
        model = GoalConditionedTransformer(args=args)
    if args.gpu:
        model = model.to('cuda')
    all_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(all_parameters, lr=args.lr)

    # initial evaluation
    best_acc = 0.0
    best_loss = float("inf")
    # """
    model.eval()
    # save_predictions
    pred_save_filename = args.save_path.split('.')[0]+'.jsonl'
    with torch.no_grad():
        eval_iter = tqdm(dl_splits['valid_seen'], desc='valid (seen)')
        best_acc = 0.0
        best_loss = 0.0
        n_batches = 0
        json_outputs = []
        for batch, feat in eval_iter:
            # 'input_goals', 'state_seq', 'action_seq', 'action_mask_seq'
            all_outputs = model.train_forward(
                goal_representation=feat['input_goals']['input_ids'],
                action_sequence=feat['action_seq_w_curr']['input_ids'],
                image_sequence=feat['state_seq_w_curr'],
                i_mask=feat['input_goals']['attention_mask'],
                o_mask=feat['action_seq_w_curr']['attention_mask'],
                label_smoothing=args.label_smoothing,
            )
            loss = all_outputs.loss
            outputs = model.test_generate(
                goal_representation=feat['input_goals']['input_ids'],
                action_seq_past=feat['action_seq_past']['input_ids'],
                image_seq_w_curr=feat['state_seq_w_curr'],
                i_mask=feat['input_goals']['attention_mask'],
                o_mask=feat['action_seq_past']['attention_mask'],
            )['actions']
            all_action_scores, new_action_states = model.score_all_continuations(
                goal_representation=feat['input_goals']['input_ids'],
                action_seq_past=feat['action_seq_past']['input_ids'],
                image_seq_w_curr=feat['state_seq_w_curr'],
                i_mask=feat['input_goals']['attention_mask'],
                o_mask=feat['action_seq_past']['attention_mask'],
                continuations = [x + ":" for x in API_ACTIONS_SN[:8]] + [x + "," for x in API_ACTIONS_SN[8:]]
            )
            is_interact_action = all_action_scores.argmax(-1) < 8
            best_actions_fullscore = new_action_states['actions']
            if is_interact_action.any():
                # TODO replicate final state once
                new_image_seq = torch.cat([new_action_states['states_seq'], new_action_states['states_seq'][:,-1,:].unsqueeze(1)], dim=1)
                new_image_seq = new_image_seq.clone()
                last_pos_idx = new_action_states['action_seq_mask'].sum(1)
                new_image_seq[:,last_pos_idx,:] = new_image_seq[:,last_pos_idx-1,:]
                bs = feat['input_goals']['input_ids'].size(0)

                # score actions
                all_action_scores_objs, new_action_states_objs = model.score_all_continuations(
                    goal_representation=feat['input_goals']['input_ids'],
                    action_seq_past=new_action_states['action_seq'],
                    image_seq_w_curr=new_image_seq,
                    i_mask=feat['input_goals']['attention_mask'],
                    o_mask=new_action_states['action_seq_mask'],
                    continuations = [obj+',' for obj in CLASSES_NATURALIZED],
                )
                last_token_pos = (new_action_states['actions'] != 0).sum(-1)
                for interact_idx in is_interact_action.nonzero():
                    interact_idx = interact_idx.squeeze()
                    total_action_size = new_action_states_objs['actions'].size(1) + last_token_pos[interact_idx]
                    if best_actions_fullscore.size(1) < total_action_size:
                        best_actions_fullscore = F.pad(best_actions_fullscore, pad=(0,total_action_size-best_actions_fullscore.size(1),0,0), value=0)
                    best_actions_fullscore[
                        interact_idx, last_token_pos[interact_idx]:total_action_size
                    ] = new_action_states_objs['actions'][interact_idx]  # TODO check if changed
            metrics = dl_splits['valid_seen'].compute_metrics(outputs, feat)
            # metrics = dl_splits['valid_seen'].compute_metrics(best_actions_fullscore, feat)
            acc = metrics['accuracy']
            outputs = metrics['output_dicts']
            for idx, output in enumerate(outputs):
                output['inputs'] = model.tokenizer.decode(feat['input_goals']['input_ids'][idx], skip_special_tokens=True)
                output['action_history'] = model.tokenizer.decode(feat['action_seq_past']['input_ids'][idx], skip_special_tokens=True)
            best_acc += acc
            best_loss += loss
            n_batches += 1
            eval_iter.set_description(f"valid (seen) loss: {best_loss / n_batches} // accuracy: {best_acc / n_batches}")
            json_outputs += outputs
        best_acc = best_acc / n_batches if n_batches > 0 else 0
        best_loss = best_loss / n_batches if n_batches > 0 else 0
        print(f"Initial valid (seen) loss: {best_loss} // accuracy: {best_acc}")

    with open(pred_save_filename, 'w') as wf:
        for json_output in json_outputs:
            wf.write(json.dumps(json_output)+"\n")
    # """

    # training loop
    for epoch in range(args.epoch):
        print(f"EPOCH {epoch}")

        # train
        model.train()
        train_iter = tqdm(dl_splits['train'], desc='training loss')
        step = 0
        for batch, feat in train_iter:
            optimizer.zero_grad()
            # 'input_goals', 'state_seq', 'action_seq', 'action_mask_seq'
            outputs = model.train_forward(
                goal_representation=feat['input_goals']['input_ids'],
                action_sequence=feat['action_seq_w_curr']['input_ids'],
                image_sequence=feat['state_seq_w_curr'],
                i_mask=feat['input_goals']['attention_mask'],
                o_mask=feat['action_seq_w_curr']['attention_mask'],
                label_smoothing=args.label_smoothing,
            )
            # output = model(input_ids=feat['input_goals']['input_ids'], attention_mask=feat['input_goals']['attention_mask'], =tgt_action, return_dict=True)
            # feat['frames'])
            loss = outputs.loss.mean()
            train_iter.set_description(f"training loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            step += 1
            if step%2000 == 0:
                torch.save(model.state_dict(), f"{save_path[:-4]}_ep{epoch}_step{step}.pth")
                # evaluate
                model.eval()
                with torch.no_grad():
                    eval_iter = tqdm(dl_splits['valid_seen'], desc='valid (seen)')
                    epoch_acc = 0.0
                    epoch_loss = 0.0
                    n_batches = 0
                    for batch, feat in eval_iter:
                        # 'input_goals', 'state_seq', 'action_seq', 'action_mask_seq'
                        loss = model.train_forward(
                            goal_representation=feat['input_goals']['input_ids'],
                            action_sequence=feat['action_seq_w_curr']['input_ids'],
                            image_sequence=feat['state_seq_w_curr'],
                            i_mask=feat['input_goals']['attention_mask'],
                            o_mask=feat['action_seq_w_curr']['attention_mask'],
                        ).loss
                        outputs = model.test_generate(
                            goal_representation=feat['input_goals']['input_ids'],
                            action_seq_past=feat['action_seq_past']['input_ids'],
                            image_seq_w_curr=feat['state_seq_w_curr'],
                            i_mask=feat['input_goals']['attention_mask'],
                            o_mask=feat['action_seq_past']['attention_mask'],
                        )['actions']
                        metrics = dl_splits['valid_seen'].compute_metrics(outputs, feat)
                        acc = metrics['accuracy']
                        epoch_loss += loss
                        epoch_acc += acc
                        n_batches += 1
                        eval_iter.set_description(f"valid (seen) loss: {epoch_loss / n_batches} // accuracy: {epoch_acc / n_batches}")
                    epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0
                    epoch_acc = epoch_acc / n_batches if n_batches > 0 else 0
                    print(f"Epoch {epoch} valid (seen) loss: {epoch_loss} // accuracy: {epoch_acc}")
                    if epoch_acc > best_acc  or (epoch_acc == best_acc and epoch_loss < best_loss):
                        print("Saving model")
                        torch.save(model.state_dict(), save_path)
                        best_loss = epoch_loss
                        best_acc = epoch_acc

        # # evaluate
        # model.eval()
        # with torch.no_grad():
        #     eval_iter = tqdm(dl_splits['valid_seen'], desc='valid (seen)')
        #     epoch_acc = 0.0
        #     epoch_loss = 0.0
        #     n_batches = 0
        #     for batch, feat in eval_iter:
        #         # 'input_goals', 'state_seq', 'action_seq', 'action_mask_seq'
        #         #TODO
        #         # 'input_goals', 'state_seq', 'action_seq', 'action_mask_seq'
        #         loss = model.train_forward(
        #             goal_representation=feat['input_goals']['input_ids'],
        #             action_sequence=feat['action_seq_w_curr']['input_ids'],
        #             image_sequence=feat['state_seq_w_curr'],
        #             i_mask=feat['input_goals']['attention_mask'],
        #             o_mask=feat['action_seq_w_curr']['attention_mask'],
        #         ).loss
        #         outputs, _ = model.test_generate(
        #             goal_representation=feat['input_goals']['input_ids'],
        #             action_seq_past=feat['action_seq_past']['input_ids'],
        #             image_seq_w_curr=feat['state_seq_w_curr'],
        #             i_mask=feat['input_goals']['attention_mask'],
        #             o_mask=feat['action_seq_past']['attention_mask'],
        #         )['actions']
        #         acc = dl_splits['valid_seen'].compute_metrics(outputs, feat)['accuracy']
        #         epoch_loss += loss
        #         epoch_acc += acc
        #         n_batches += 1
        #         eval_iter.set_description(f"valid (seen) loss: {epoch_loss / n_batches} // accuracy: {epoch_acc / n_batches}")
        #     epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0
        #     epoch_acc = epoch_acc / n_batches if n_batches > 0 else 0
        #     print(f"Epoch {epoch} valid (seen) loss: {epoch_loss} // accuracy: {epoch_acc}")
        #     if epoch_acc > best_acc  or (epoch_acc == best_acc and epoch_loss < best_loss):
        #         print("Saving model")
        #         torch.save(model.state_dict(), save_path)
        #         best_loss = epoch_loss
        #         best_acc = epoch_acc
        # torch.save(model.state_dict(), f"{save_path[:-4]}_ep{epoch}.pth")
