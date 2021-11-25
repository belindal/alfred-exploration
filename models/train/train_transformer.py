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

from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import trange
import collections
import numpy as np
from gen.utils.image_util import decompress_mask
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW
from models.model.t5 import GoalConditionedTransformer
from tqdm import tqdm


# python models/train/train_transformer.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
class ALFREDDataloader(DataLoader):
    def __init__(self, args, vocab, dataset, split_type, batch_size, sep_actions: bool):
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
        self.dataset = [self.load_task_json(task, args.data, args.pp_folder) for task in dataset]

        if "train" in split_type or "valid" in split_type:
            # self.dataset = self.featurize(self.dataset)
            new_dataset = []
            for ex in tqdm(self.dataset):
                goal = self.vocab['word'].index2word(ex['num']['lang_goal'])

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

                assert len(ex['num']['lang_instr']) == len(ex['num']['action_low'])
                prev_subgoals = goal
                prev_actions = []
                prev_action_masks = []
                action_idx = 0
                for subgoal_idx, subgoal in enumerate(ex['num']['lang_instr']):
                    subgoal = self.vocab['word'].index2word(subgoal)
                    # in terms of the action vocabulary
                    # subgoal_actions = self.vocab['action_high'].index2word(ex['num']['action_high'][subgoal_idx]['action'])
                    for action in ex['num']['action_low'][subgoal_idx]:
                        action_low = self.vocab['action_low'].index2word(action['action'])
                        # unravel actions
                        if sep_actions:
                            new_ex = {
                                'goal': goal,
                                'curr_subgoal': subgoal,
                                'prev_subgoals': [sg for sg in prev_subgoals],
                                'curr_state': all_frames[action_idx],
                                'prev_actions': [a for a in prev_actions],
                                'curr_action': action_low,
                            }
                            if action['mask'] is not None: new_ex['curr_action_mask'] = self.decompress_mask(action['mask'])
                            new_dataset.append(new_ex)
                        if action['mask'] is not None: prev_action_masks.append(self.decompress_mask(action['mask']))
                        prev_actions.append(action_low)
                        action_idx += 1
                    if not sep_actions:
                        new_ex = {
                            'goal': goal,
                            'curr_subgoal': subgoal,
                            'prev_subgoals': [sg for sg in prev_subgoals],
                            'all_states': all_frames,
                            'all_actions': [a for a in prev_actions],
                        }
                        if len(prev_action_masks) > 0: new_ex['all_actions_mask'] = prev_action_masks
                        new_dataset.append(new_ex)
                    prev_subgoals += subgoal
                self.dataset = new_dataset
        super().__init__(self.dataset, batch_size, collate_fn=self.collate_fn)


    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(mask)

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def serialize_lang_action(self, feat):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            # if not self.test_mode:
            feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########
            # subgoal completion supervision
            if self.args.subgoal_aux_loss_wt > 0:
                feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)

            # progress monitor supervision
            if self.args.pm_aux_loss_wt > 0:
                num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########
            # serialize segments
            self.serialize_lang_action(ex)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # # zero inputs if specified
            # lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            # lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            lang_goal_instr = lang_goal + lang_instr
            feat['lang_goal_instr'].append(lang_goal_instr)

            # load Resnet features from disk
            if load_frames:   #and not self.test_mode:
                root = self.get_task_root(ex)
                im = torch.load(os.path.join(root, self.feat_pt))

                num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action
                num_feat_frames = im.shape[0]

                # Modeling Quickstart (without filler frames)
                if num_low_actions == num_feat_frames:
                    feat['frames'].append(im)

                # Full Dataset (contains filler frames)
                else:
                    keep = [None] * num_low_actions
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep[-1] = im[-1]  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))

            #########
            # outputs
            #########
            # low-level action
            feat['action_low'].append([a['action'] for a in ex['num']['action_low']])

            # low-level action mask
            if load_mask:
                feat['action_low_mask'].append([self.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None])

            # low-level valid interact
            feat['action_low_valid_interact'].append([a['valid_interact'] for a in ex['num']['action_low']])

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr', 'action_low', 'action_high'}:
                # language embedding and padding
                seqs = [' '.join(self.vocab['word'].index2word(vv)) for vv in v]
                # seqs = self.tokenizer(seqs, return_tensors='pt', padding=True).to(device)
                # pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                # seq_lengths = np.array(list(map(len, v)))
                # # TODO self.vocab
                # embed_seq = self.emb_word(pad_seq)
                # packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = seqs
            elif k in {'action_low_mask'}:
                continue
                # seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                # feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(
                    vv, #device=device,
                    dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(
                    vv, #device=device,
                    dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat

    def load_task_json(self, task, data_path, pp_folder):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(data_path, task['task'], '%s' % pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def pad_stack(self, data: list):
        max_item_len = max([len(item) for item in data])
        data = [torch.cat([
            item, torch.zeros(max_item_len - item.size(0), *item.size()[1:]).to(item.device, item.dtype)
        ], dim=0) for item in data]
        return torch.stack(data)

    def collate_fn(self, batch):
        # for i in trange(0, len(data), batch_size, desc='batch'):
        # tasks = data[i:i+batch_size]
        # batch = [self.load_task_json(task) for task in batch]
        # for task in batch:
        #     assert len(task['ann']['instr']) == len(task['plan']['high_pddl'])
        # feat = self.featurize(batch)
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        if 'prev_actions' in batch[0]:
            feat = {
                'input_goals': [],
                'curr_state': [],
                'prev_actions': [],
                'tgt_action': [],
                'tgt_action_mask': [],
            }
        else:
            feat = {
                'input_goals': [],
                'all_states': [],
                'all_actions': [],
                'all_actions_mask': [],
            }
        for item in batch:
            # 'goal', 'curr_subgoal', 'prev_subgoals', 'curr_state', 'prev_actions', 'curr_action', 'curr_action_mask'
            # 'goal', 'curr_subgoal', 'prev_subgoals', 'all_states', 'all_actions', 'all_actions_mask'
            feat['input_goals'].append(' '.join(item['prev_subgoals'] + item['curr_subgoal']))
            if 'prev_actions' in item:
                feat['curr_state'].append(item['curr_state'].to(device))
                feat['prev_actions'].append(' '.join(item['prev_actions']))
                feat['tgt_action'].append(' '.join(item['curr_action']))
                if 'curr_action_mask' in item:
                    feat['tgt_action_mask'].append(item['curr_action_mask'].to(device))
            else:
                assert 'all_actions' in item
                # expand `all_states` for each token of actions
                all_states_expanded = []
                # TODO unroll `actions_mask` with this
                for action_idx in range(len(item['all_actions'])):
                    for _ in self.tokenizer.tokenize(item['all_actions'][action_idx]):
                        all_states_expanded.append(item['all_states'][action_idx])
                assert len(all_states_expanded) == len(self.tokenizer.tokenize(' '.join(item['all_actions'])))
                # (n_actions, 512, width, height)
                feat['all_states'].append(torch.stack(all_states_expanded).to(device))
                feat['all_actions'].append(' '.join(item['all_actions']))
                if 'all_actions_mask' in item:
                    feat['all_actions_mask'].append(
                        torch.cat([mask.to(device) for mask in item['all_actions_mask']])
                    )
        for key in feat:
            if 'state' not in key and 'mask' not in key:
                feat[key] = self.tokenizer(feat[key], return_tensors='pt', padding=True).to(device)
            else:
                feat[key] = self.pad_stack(feat[key])
        return batch, feat

    # def iterate(self, data, batch_size):
    #     '''
    #     breaks dataset into batch_size chunks for training
    #     '''
    #     for i in trange(0, len(data), batch_size, desc='batch'):
    #         tasks = data[i:i+batch_size]
    #         batch = [self.load_task_json(task) for task in tasks]
    #         feat = self.featurize(batch)
    #         yield batch, feat


def compute_metric(self, preds, data):
    '''
    compute f1 and extract match scores for output
    '''
    m = collections.defaultdict(list)
    for task in data:
        i = self.get_task_and_ann_id(ex)
        label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
        m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
        m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
    return {k: sum(v)/len(v) for k, v in m.items()}


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_im')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')
    parser.add_argument('--use_templated_goals', help='use templated goals instead of human-annotated goal descriptions (only available for train set)', action='store_true')
    parser.add_argument('--resume', help='load a checkpoint')

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
        sep_actions = False
        print(f"Processing {split_type} dataset")
        if args.fast_epoch:
            splits[split_type] = splits[split_type][:10]
        else:
            splits[split_type] = splits[split_type][:1000]
        dl_splits[split_type] = ALFREDDataloader(args, vocab, splits[split_type], split_type, args.batch, sep_actions)

    # load model
    # model = T5ForConditionalGeneration.from_pretrained('t5-small').to('cuda')
    model = GoalConditionedTransformer()
    if args.gpu:
        model = model.to('cuda')
    all_parameters = [p for p in model.parameters() if p.requires_grad]
    # TODO ADAMW????
    # optimizer = AdamW(all_parameters, lr=args.lr)
    optimizer = torch.optim.Adam(all_parameters, lr=args.lr)
    for epoch in range(args.epoch):
        print(f"EPOCH {epoch}")
        pbar = tqdm(dl_splits['train'], desc='batch')
        for batch, feat in pbar:
            optimizer.zero_grad()
            # 'input_goals', 'curr_state', 'prev_actions', 'curr_action', 'curr_action_mask'
            # 'input_goals', 'all_states', 'all_actions', 'all_actions_mask'
            # encoder_outputs: ModelOutput = encoder(model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'], return_dict=True)
            # encoder_outs = model.get_encoder()(input_ids=feat['input_goals']['input_ids'], attention_mask=feat['input_goals']['attention_mask'], return_dict=True)
            # encoder_outs = model.get_decoder()(encoder_outputs=encoder_outs)
            outputs = model.train_forward(
                goal_representation=feat['input_goals']['input_ids'],
                action_sequence=feat['all_actions']['input_ids'],
                image_sequence=feat['all_states'],
                i_mask=feat['input_goals']['attention_mask'],
                o_mask=feat['all_actions']['attention_mask'],
            )
            # output = model(input_ids=feat['input_goals']['input_ids'], attention_mask=feat['input_goals']['attention_mask'], =tgt_action, return_dict=True)
            # feat['frames'])
            loss = outputs.loss.mean()
            pbar.set_description(f"training loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        # evaluate
        for batch, feat in dl_splits['valid_seen']:
            #TODO
            continue
