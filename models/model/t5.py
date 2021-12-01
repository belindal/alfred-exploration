import os
import torch
import numpy as np
import json
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F
from models.nn.vnn import ResnetVisualEncoder
import regex as re
import gen


def unCamelSnakeCase(action_str):
    """
    for all actions and all objects uncamel case and unsnake case.
    Also remove all nubmers
    """
    return re.sub(r'((?=[A-Z])|(\d+)|_|  )', ' ', action_str).lower().strip()

def snake_to_camel(action_str):
    """
    for all actions and all objects unsnake case and camel case.
    re-add numbers
    """
    def camel(match):
        return match.group(1)[0].upper() + match.group(1)[1:] + match.group(2).upper()
    action_str = re.sub(r'(.*?) ([a-zA-Z])', camel, action_str)
    if action_str.startswith("Look"):  # LookDown_15, LookUp_15
        action_str += "_15"
    if action_str.startswith("Rotate"):  # RotateRight_90, RotateLeft_90
        action_str += "_90"
    if action_str.startswith("Move"):  # MoveAhead_25
        action_str += "_25"
    return action_str[0].upper() + action_str[1:]

vis_encoder = ResnetVisualEncoder(dframe=512)
API_ACTIONS = ["PickupObject", "ToggleObject", "LookDown_15", "MoveAhead_25", "RotateLeft_90", "LookUp_15", "RotateRight_90", "ToggleObjectOn", "ToggleObjectOff", "PutObject", "SliceObject", "OpenObject", "CloseObject"]
API_ACTIONS_NATURALIZED = [unCamelSnakeCase(action) for action in API_ACTIONS]
CLASSES = ['0'] + gen.constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp',
                                        'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']

class GoalConditionedTransformer(nn.Module):
    def __init__(self, concat_dim=1024, hidden_dim=512,random_init=0, args=None):
        super(GoalConditionedTransformer, self).__init__()
        self.args = args
        if random_init==1:
            config = AutoConfig.from_pretrained('t5-small')
            self.model = AutoModelForSeq2SeqLM.from_config(config)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.concat_dim = concat_dim
        self.hidden_dim = hidden_dim
        self.fusion_module = nn.Linear(self.concat_dim, self.hidden_dim)
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

    def setup_inputs_for_train(self, goal_representation, action_sequence, image_sequence,i_mask,o_mask):
        # encode goal with T5 encoder
        # assume action_sequence doesn't have start id
        # (bs, n_tokens,)
        action_sequence_shifted = action_sequence.new_zeros(action_sequence.shape)
        # action_sequence_shifted[..., 1:] = action_sequence_shifted[..., :-1].clone()
        action_sequence_shifted[..., 1:] = action_sequence[..., :-1].clone()
        action_sequence_shifted[..., 0] = self.model.config.decoder_start_token_id

        # (bs, n_tokens, word_embed_dim)
        embedded_action_sequence = self.model.decoder.embed_tokens(action_sequence_shifted)
        # (bs, n_tokens, word_embed_dim + image_dim) -> (bs, n_tokens, fusion_output_dim)
        fused_action_image_rep = self.fusion_module(torch.cat([embedded_action_sequence, image_sequence], dim=-1))

        labels = action_sequence.clone()
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': goal_representation,
            'attention_mask': i_mask,
            'decoder_inputs_embeds': fused_action_image_rep,
            'decoder_attention_mask': o_mask,
            'labels': labels,
        }

    def train_forward(self, goal_representation, action_sequence, image_sequence,i_mask,o_mask):
        transformer_inputs = self.setup_inputs_for_train(goal_representation, action_sequence, image_sequence,i_mask,o_mask)
        model_outs = self.model(**transformer_inputs)
        return model_outs

    def setup_inputs_for_generate(self, goal_representation, action_seq_past, image_seq_w_curr, i_mask=None, o_mask=None):
        """
        Suppose action 0 has 4 tokens: [a0, a0, a0, a0]
        [bos, a0, a0, a0, a0] = action_sequence
        [s0,  s0, s0, s0, s1] = image_sequence
        """
        # (bs, n_tokens_in_history + 1)
        action_sequence = F.pad(input=action_seq_past, pad=(1,0,0,0), value=self.model.config.decoder_start_token_id)  # prepend bos token
        action_sequence_mask = F.pad(input=o_mask, pad=(1,0,0,0), value=1)
        image_sequence = image_seq_w_curr[:,:action_seq_past.size(1)+1,:].clone()
        assert image_sequence.size()[:2] == action_sequence.size()
        # # sanity check
        # if image_sequence.size(1) > 1:
        #     assert not (image_sequence[:,-1,:] == image_sequence[:,-2,:]).all()
        embedded_action_sequence = self.model.decoder.embed_tokens(action_sequence)
        # (bs, n_tokens_in_history + 1, linear_out_dim)
        fused_action_image_rep = self.fusion_module(torch.cat([embedded_action_sequence, image_sequence], dim=-1))
        return action_sequence, action_sequence_mask, image_sequence, fused_action_image_rep

    def test_generate(self, goal_representation, action_seq_past, image_seq_w_curr, i_mask=None, o_mask=None, topk=1):
        """
        goal_representation: (bs, # tokens in goal)
        action_seq_past: (bs, tok_len of action sequence history)
        image_seq_w_curr: (bs, tok_len of action sequence history + tok_len of curr action/1, image_dim) [+1 for end of next state]
        i_mask: 
        """
        action_sequence, action_sequence_mask, image_sequence, fused_action_image_rep = self.setup_inputs_for_generate(
            goal_representation, action_seq_past, image_seq_w_curr, i_mask=i_mask, o_mask=o_mask,
        )

        scores = []
        next_actions = []
        bs = goal_representation.size(0)
        # greedy search
        # pad all tok_len dimensions by 10
        # last token = first `0` token of mask
        action_sequence = F.pad(action_sequence, pad=(0,15,0,0), value=self.tokenizer.pad_token_id)
        image_sequence = F.pad(image_sequence, pad=(0,0,0,15,0,0), value=0)
        fused_action_image_rep = F.pad(fused_action_image_rep, pad=(0,0,0,15,0,0), value=0)
        action_sequence_mask = F.pad(action_sequence_mask, pad=(0,15,0,0), value=0)
        last_token_pos = torch.stack([
            (action_sequence[idx] == 6).nonzero().max()
            if (action_sequence[idx] == 6).any() else torch.tensor(0).to(self.device)
            for idx in range(bs)
        ])
        ended_actions = torch.zeros(bs).bool().to(self.device)
        for unroll_idx in range(15):
            model_output = self.model(
                input_ids=goal_representation, attention_mask=i_mask,
                decoder_inputs_embeds=fused_action_image_rep, decoder_attention_mask=action_sequence_mask,
            )
            if unroll_idx == 0:
                next_logit_score_dist, next_logits_dist = model_output.logits[torch.arange(bs),last_token_pos].topk(topk, dim=-1)
                scores_dist = torch.distributions.Categorical(logits = next_logit_score_dist)
                sampled_action_idx = scores_dist.sample()
                next_logit_scores, next_logits = next_logit_score_dist[torch.arange(bs), sampled_action_idx], next_logits_dist[torch.arange(bs),sampled_action_idx]
            else:
                next_logit_scores, next_logits = model_output.logits[torch.arange(bs),last_token_pos].max(-1)
            next_image = image_sequence[torch.arange(bs),last_token_pos]  # repeat current state
            # if the action has ended, next token must be padding
            next_logit_scores[ended_actions] = 0  # P(pad after end) = 1
            next_logits[ended_actions] = self.tokenizer.pad_token_id
            next_image[ended_actions] = 0
            # [(bs) x n_gen_tokens]
            scores.append(next_logit_scores)
            next_actions.append(next_logits)
            # sequence has expanded
            last_token_pos += 1
            # (bs, n_tokens+1)
            action_sequence[torch.arange(bs),last_token_pos] = next_logits
            embedded_action_sequence = self.model.decoder.embed_tokens(action_sequence)
            # (bs, n_tokens+1, image_dim)
            image_sequence[torch.arange(bs),last_token_pos] = next_image
            fused_action_image_rep = self.fusion_module(torch.cat([embedded_action_sequence, image_sequence], dim=-1))
            # (bs, n_tokens+1)
            action_sequence_mask[torch.arange(bs),last_token_pos] = 1
            # """
            # break on comma -- next action
            ended_actions |= next_logits == self.tokenizer.convert_tokens_to_ids(',')
            if ended_actions.all(): break
            # """
        if ended_actions.all():
            # remove extra padding and bos token id
            # (bs, n_tokens)
            action_sequence = action_sequence[:,1:last_token_pos.max()+1]
            action_sequence_mask = action_sequence_mask[:,1:last_token_pos.max()+1]
            # (bs, n_tokens, image_dim)
            image_sequence = image_sequence[:,1:last_token_pos.max()+1,:]
        # bs x n_gen_tokens
        next_actions = torch.stack(next_actions, dim=1)
        # bs x n_gen_tokens -> bs
        scores = -torch.stack(scores, dim=1).sum(-1)
        return {'actions': next_actions, 'action_seq': action_sequence, 'action_seq_mask': action_sequence_mask, 'states_seq': image_sequence, 'log_probs': scores}

    def score_all_actions(self, goal_representation, action_seq_past, image_seq_w_curr, i_mask, o_mask):
        action_sequence, action_sequence_mask, image_sequence, fused_action_image_rep = self.setup_inputs_for_generate(
            goal_representation, action_seq_past, image_seq_w_curr, i_mask=i_mask, o_mask=o_mask,
        )
        # return scores of all possible actions
        encoder_outputs = self.model.encoder(
            input_ids=goal_representation, attention_mask=i_mask,
        )
        all_action_tokens = self.tokenizer(API_ACTIONS_NATURALIZED, return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)

        bs = goal_representation.size(0)
        last_token_pos = torch.stack([
            (action_sequence[idx] == 6).nonzero().max() if (action_sequence[idx] == 6).any() else torch.tensor(0).to(self.device) for idx in range(bs)
        ])
        breakpoint()
        action_scores = []
        # iterate across tokens of each action
        for tok_idx in range(all_action_tokens['input_ids'].size(1)):
            model_outputs = self.model(
                encoder_outputs=encoder_outputs, decoder_inputs_embeds=fused_action_image_rep, decoder_attention_mask=action_sequence_mask,
            )
            # (bs, seqlen, vocab_size) -> (bs, vocab_size)
            last_tok_logits = model_outputs.logits[torch.arange(bs), last_token_pos]
            last_tok_probs = torch.log_softmax(last_tok_logits, dim=-1)
            # (bs, vocab_size) -> (bs, # actions, vocab_size)
            batch_action_probs = last_tok_probs[:, all_action_tokens['input_ids'][:,tok_idx]]
    
    @classmethod
    def load(cls, args, fsave):
        model = cls(args=args)
        if args.gpu:
            model.load_state_dict(torch.load(fsave))
        else:
            model.load_state_dict(torch.load(fsave, map_location=torch.device('cpu')))
        return model

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def featurize(self, batch):
        feat = {
            # 'all_states': [],
            # 'actions': [],
            # 'actions_mask': [],
            "goal_representation": [],
        }
        for item in batch:
            # dict_keys(['ann', 'images', 'num', 'pddl_params', 'plan', 'repeat_idx', 'root', 'scene', 'split', 'task_id', 'task_type', 'turk_annotations'])
            feat['goal_representation'].append(
                ''.join([g.rstrip() for g in item['ann']['goal']]).replace('<<goal>>', ' [goal]').replace('<<stop>>', ' [stop]').replace('  ', ' ').strip()
            )
            # TODO: trim <<goal>> and <<stop>> from goal rep?
            for instr in item['ann']['instr']:
                feat['goal_representation'][-1] += (
                    ''.join([i.rstrip() for i in instr])
                ).replace('<<goal>>', ' [goal]').replace('<<stop>>', ' [stop]').replace('  ', ' ').strip()
        feat['goal_representation'] = self.tokenizer(feat['goal_representation'], return_tensors='pt', padding=True).to(self.device)
        feat['actions'] = {
            'input_ids': torch.Tensor(len(batch),0).long().to(self.device),
            'attention_mask': torch.Tensor(len(batch),0).long().to(self.device),
        }
        return feat

    @classmethod
    def generate_naive_action_mask(cls, _action, _curr_image, _curr_image_features):
        m = np.zeros((300, 300))
        m[140:160, 140:160] = 1
        return m

    @classmethod
    def generate_action_mask(cls, action, curr_image, image_obj_features):
        m = np.zeros((300, 300))
        # extract object(s) from action
        object_names = action.split(':')[-1].strip()
        object_names = object_names.split(' in ')
        if len(object_names) == 0: return None
        # ["table", "chair"]
        obj_label2feature_idx = {CLASSES[label]: i for i, label in enumerate(image_obj_features["class_labels"])}
        obj_masks = []
        for obj_name in object_names:
            try:
                obj_idx = obj_label2feature_idx[snake_to_camel(obj_name)]
            except Exception as e:
                print("trying to interact with an object not in the scene")
                breakpoint()
            obj_masks.append(image_obj_features['masks'][obj_idx][0])
        return m

    def decode_prediction(self, m_out, curr_image, object_features):
        # TODO: Maybe cast all actions to be within the 13 valid tokens.
        # Also this is a good place to add in the exploration
        action = self.tokenizer.decode(m_out, skip_special_tokens=True).split(",")[0].strip()
        # convert BACK to API action!!!
        api_action = snake_to_camel(action.split(':')[0].strip())
        assert api_action in API_ACTIONS, f"{api_action} is not part of {API_ACTIONS}!"
        mask = (
            self.generate_action_mask(action, curr_image, object_features)
            if self.has_interaction(action)
            else None
        )
        return {
            "action_low": api_action,
            "action_low_mask": mask,
        }

    @classmethod
    def has_interaction(cls, action):
        """
        check if low-level action is interactive
        """
        return ':' in action
        # non_interact_actions = [
        #     "MoveAhead",
        #     "Rotate",
        #     "Look",
        #     "[stop]",
        #     "<<pad>>",
        #     "<<seg>>",
        # ]
        # if any(a in action for a in non_interact_actions):
        #     return False
        # else:
        #     return True
