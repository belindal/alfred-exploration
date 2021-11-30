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
from models.model.t5 import GoalConditionedTransformer
from tqdm import tqdm
import itertools as it
import random
random.seed(0)


# python models/train/train_transformer.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
class ALFREDDataloader(DataLoader):
    """
    *_seq_past: previous states/actions before the curr state (i.e. [a,b,c])
    *_curr; curr (single) state/action to output (i.e. d)
    *_seq_w_curr: sequence of states/actions including the curr one (i.e. [a,b,c,d])
    """
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
        self.action_mask_dim = None

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
                action_mask_sequence = []
                all_subgoals = [[sg.rstrip() for sg in subgoal] for subgoal in ex['ann']['instr']]
                all_subgoals_flattened = list(it.chain(*all_subgoals))
                action_idx = 0
                for subgoal_idx, subgoal in enumerate(all_subgoals):
                    # in terms of the action vocabulary
                    # subgoal_actions = self.vocab['action_high'].index2word(ex['num']['action_high'][subgoal_idx]['action'])
                    for action in ex['num']['action_low'][subgoal_idx]:
                        action_low = self.vocab['action_low'].index2word(action['action'])
                        action_sequence.append(action_low)
                        if action['mask'] is not None:
                            assert action['valid_interact'] == 1
                        else:
                            assert action['valid_interact'] == 0
                        curr_action_mask = self.decompress_mask(action['mask']) if action['mask'] is not None else None
                        if curr_action_mask is not None and self.action_mask_dim is None:
                            self.action_mask_dim = curr_action_mask.size()
                        action_mask_sequence.append(curr_action_mask)
                        action_idx += 1

                        new_ex = {
                            'goal': goal,
                            # make copies of lists
                            'all_subgoals': all_subgoals_flattened, 'prev_subgoals': [sg for sg in prev_subgoals], 'curr_subgoal': subgoal,
                            'state_seq': all_frames[:action_idx], 'action_seq': [a for a in action_sequence], 'action_mask_seq': [am for am in action_mask_sequence],
                        }
                        assert len(new_ex['state_seq']) == len(new_ex['action_seq']) == len(new_ex['action_mask_seq'])
                        new_dataset.append(new_ex)
                    prev_subgoals += subgoal
        else:
            new_dataset = []
            for task in tqdm(dataset):
                new_dataset.append(self.load_task_json(task, args.data, args.pp_folder))
        super().__init__(new_dataset, batch_size, collate_fn=self.collate_fn)


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
            assert len(item['action_seq']) == len(item['state_seq']) == len(item['action_mask_seq'])
            # 'goal', 'curr_subgoal', 'prev_subgoals', 'all_subgoals', '{state|action|action_mask}_{history|next|seq}',
            # if gt_alignment:
            #     feat['input_goals'].append(' '.join(item['goal'] + item['prev_subgoals'] + item['curr_subgoal']))
            feat['input_goals'].append(''.join(item['goal'] + item['all_subgoals']).replace('  ', ' ').strip())

            # expand `state_seq` and `action_mask_seq` for each token of `action_seq`
            expanded_seqs = {f'{seq_type}_{seq_span}': [] for seq_type in ['state', 'action_mask'] for seq_span in ['seq_w_curr', 'seq_past', 'curr']}
            for action_idx in range(len(item['action_seq'])):
                for _ in self.tokenizer.tokenize(item['action_seq'][action_idx]):
                    for seq_type in ['state', 'action_mask']:
                        if item[f"{seq_type}_seq"][action_idx] is None:
                            assert seq_type == 'action_mask'
                            item[f"{seq_type}_seq"][action_idx] = torch.full(self.action_mask_dim, -100)  # (1, 300, 300)
                        expanded_seqs[f'{seq_type}_seq_past' if action_idx < len(item['action_seq']) - 1 else f'{seq_type}_curr'].append(item[f"{seq_type}_seq"][action_idx])
                        expanded_seqs[f'{seq_type}_seq_w_curr'].append(item[f"{seq_type}_seq"][action_idx])
            for seq_key in expanded_seqs.keys():
                feat[seq_key].append(
                    torch.stack(expanded_seqs[seq_key]).to(device)
                    if len(expanded_seqs[seq_key]) > 0 else torch.tensor([]).to(device)
                )

            feat['action_seq_w_curr'].append(' '.join(item['action_seq']))
            feat['action_seq_past'].append(' '.join(item['action_seq'][:-1]))
            feat['action_curr'].append(item['action_seq'][-1])

            # (n_actions, 512)
            assert feat['state_seq_w_curr'][-1].size(0) == len(self.tokenizer.tokenize(feat['action_seq_w_curr'][-1]))
            # # (n_actions, 300, 300)
            assert feat['action_mask_seq_w_curr'][-1].size(0) == feat['state_seq_w_curr'][-1].size(0)
        for key in feat:
            if type(feat[key][0]) == str:
                feat[key] = self.tokenizer(feat[key], return_tensors='pt', padding=True).to(device)
            elif type(feat[key][0]) == torch.Tensor:
                feat[key] = self.pad_stack(feat[key], pad_id=-100 if key=='mask' else 0)
            else:
                assert False, f"features should be of type str or tensor, but got type {type(feat[key][0])} for feature {key}"
        return batch, feat

    def compute_metrics(self, preds, data):
        n_correct = 0.0
        n_total = 0
        for idx, pred in enumerate(preds):
            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            pred = pred.split(' ')[0]  # segment out first generated action
            gt = self.tokenizer.decode(data['action_curr']['input_ids'][idx], skip_special_tokens=True)
            n_correct += pred == gt
            n_total += 1
        return {'accuracy': n_correct / n_total}


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
            splits[split_type] = splits[split_type][:5]
        elif split_type == "train":
            splits[split_type] = random.sample(splits[split_type], args.train_size)
        else:
            splits[split_type] = random.sample(splits[split_type], 50)
        dl_splits[split_type] = ALFREDDataloader(args, vocab, splits[split_type], split_type, args.batch, sep_actions)

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
    with torch.no_grad():
        # eval_iter = tqdm(dl_splits['valid_seen'], desc='valid (seen)')
        eval_iter = tqdm(dl_splits['valid_seen'], desc='valid (seen)')
        best_acc = 0.0
        best_loss = 0.0
        n_batches = 0
        for batch, feat in eval_iter:
            # 'input_goals', 'state_seq', 'action_seq', 'action_mask_seq'
            #TODO
            # 'input_goals', 'state_seq', 'action_seq', 'action_mask_seq'
            all_outputs = model.train_forward(
                goal_representation=feat['input_goals']['input_ids'],
                action_sequence=feat['action_seq_w_curr']['input_ids'],
                image_sequence=feat['state_seq_w_curr'],
                i_mask=feat['input_goals']['attention_mask'],
                o_mask=feat['action_seq_w_curr']['attention_mask'],
            )
            loss = all_outputs.loss
            outputs = model.test_generate(
                goal_representation=feat['input_goals']['input_ids'],
                action_sequence=feat['action_seq_w_curr']['input_ids'],
                image_sequence=feat['state_seq_w_curr'],
                i_mask=feat['input_goals']['attention_mask'],
                o_mask=feat['action_seq_w_curr']['attention_mask'],
            )
            acc = dl_splits['valid_seen'].compute_metrics(outputs, feat)['accuracy']
            best_acc += acc
            best_loss += loss
            n_batches += 1
            eval_iter.set_description(f"valid (seen) loss: {best_loss / n_batches} // accuracy: {best_acc / n_batches}")
        best_acc = best_acc / n_batches if n_batches > 0 else 0
        best_loss = best_loss / n_batches if n_batches > 0 else 0
        print(f"Initial valid (seen) loss: {best_loss} // accuracy: {best_acc}")
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
            )
            # output = model(input_ids=feat['input_goals']['input_ids'], attention_mask=feat['input_goals']['attention_mask'], =tgt_action, return_dict=True)
            # feat['frames'])
            loss = outputs.loss.mean()
            train_iter.set_description(f"training loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            step += 1
            if step%100 == 0:
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
                #TODO
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
                    action_sequence=feat['action_seq_past']['input_ids'],
                    image_sequence=feat['state_seq_past'],
                )
                acc = dl_splits['valid_seen'].compute_metrics(outputs, feat)['accuracy']
                epoch_loss += loss
                epoch_acc += acc
                n_batches += 1
                eval_iter.set_description(f"valid (seen) loss: {loss} // accuracy: {acc}")
            epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0
            epoch_acc = epoch_acc / n_batches if n_batches > 0 else 0
            print(f"Epoch {epoch} valid (seen) loss: {epoch_loss} // accuracy: {epoch_acc}")
            if epoch_acc > best_acc  or (epoch_acc == best_acc and epoch_loss < best_loss):
                print("Saving model")
                torch.save(model.state_dict(), save_path)
                best_loss = epoch_loss
                best_acc = epoch_acc
        torch.save(model.state_dict(), f"{save_path[:-4]}_ep{epoch}.pth")
