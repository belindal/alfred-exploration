import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F
from models.nn.vnn import ResnetVisualEncoder
import regex as re
import random
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
    if action_str == "toggle object on":
        return "ToggleObjectOn"
    elif action_str == "toggle object off":
        return "ToggleObjectOff"
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
API_ACTIONS = ["PickupObject", "ToggleObject", "LookDown_15", "MoveAhead_25", "RotateLeft_90", "LookUp_15", "RotateRight_90", "ToggleObjectOn", "ToggleObjectOff", "PutObject", "SliceObject", "OpenObject", "CloseObject", '[subgoal]', '<<stop>>']
API_ACTIONS_NATURALIZED = [unCamelSnakeCase(action) for action in API_ACTIONS]
API_ACTIONS_SORTED = ["PickupObject", "ToggleObject", "ToggleObjectOn", "ToggleObjectOff", "PutObject", "SliceObject", "OpenObject", "CloseObject", "LookDown_15", "MoveAhead_25", "RotateLeft_90", "LookUp_15", "RotateRight_90", "[subgoal]", "<<stop>>"]
API_ACTIONS_SN = [unCamelSnakeCase(action) for action in API_ACTIONS_SORTED]
CLASSES = ['0'] + gen.constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp',
                                        'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']
CLASSES_NATURALIZED = [unCamelSnakeCase(obj) for obj in CLASSES]

class GoalConditionedTransformer(nn.Module):
    def __init__(self, concat_dim=1024, hidden_dim=512,random_init=0, args=None):
        super(GoalConditionedTransformer, self).__init__()
        self.args = args
        if random_init==1:
            config = AutoConfig.from_pretrained('t5-small')
            self.model = AutoModelForSeq2SeqLM.from_config(config)
        else:
            self.model=T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

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

    def train_forward(self, goal_representation, action_sequence, image_sequence,i_mask,o_mask, label_smoothing=0.0):
        transformer_inputs = self.setup_inputs_for_train(goal_representation, action_sequence, image_sequence,i_mask,o_mask)
        labels = transformer_inputs['labels']
        model_outs = self.model(**transformer_inputs)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        model_outs.loss = loss_fct(model_outs.logits.view(-1, model_outs.logits.size(-1)), labels.view(-1))
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
        try:
            assert image_sequence.size()[:2] == action_sequence.size()
        except:
            breakpoint()
        # # sanity check
        # if image_sequence.size(1) > 1:
        #     assert not (image_sequence[:,-1,:] == image_sequence[:,-2,:]).all()
        embedded_action_sequence = self.model.decoder.embed_tokens(action_sequence)
        # (bs, n_tokens_in_history + 1, linear_out_dim)
        fused_action_image_rep = self.fusion_module(torch.cat([embedded_action_sequence, image_sequence], dim=-1))
        return action_sequence, action_sequence_mask, image_sequence, fused_action_image_rep

    def test_generate(self, goal_representation, action_seq_past, image_seq_w_curr, i_mask=None, o_mask=None, topk=1, object_list=None, temperature=1.0):
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
            if object_list is not None:
                assert bs == 1
                # TODO(sahit): annotations
                token_list = self.tokenizer(object_list, add_special_tokens=False, return_tensors='pt',padding=True)['input_ids'].flatten().to(self.device)
                #action_token_list = self.tokenizer(object_list[:len(API_ACTIONS_NATURALIZED) + 2], add_special_tokens=False, return_tensors='pt',padding=True)['input_ids'].flatten().to(self.device)
                #print(self.tokenizer.decode(token_list))
                token_mask = torch.zeros(bs, self.model.config.vocab_size, dtype=torch.bool).to(self.device)
                #action_token_mask = torch.zeros(bs, self.model.config.vocab_size, dtype=torch.bool).to(self.device)
                #action_token_mask[:,action_token_list] = True
                token_mask[:,token_list] = True
            else:
                #action_token_mask = None
                token_mask = None
            if unroll_idx == 0:
                n_cands_to_sample_from = topk
            else:
                n_cands_to_sample_from = 1
            scaled_logits = model_output.logits / temperature
            if token_mask is not None:
                next_logit_score_dist, next_logits_dist = scaled_logits[0][last_token_pos][token_mask].topk(n_cands_to_sample_from, dim=-1)
            else:
                next_logit_score_dist, next_logits_dist = scaled_logits[torch.arange(bs),last_token_pos].topk(n_cands_to_sample_from, dim=-1)
            # TODO softmax here????
            scores_dist = torch.distributions.Categorical(logits = next_logit_score_dist)
            if unroll_idx == 0:
                dist_for_logging = torch.distributions.Categorical(logits = scaled_logits[0][last_token_pos][token_mask])
                entropy = dist_for_logging.entropy()
            sampled_action_idx = scores_dist.sample()
            if token_mask is not None:
                next_logit_scores, next_logits = next_logit_score_dist[sampled_action_idx].unsqueeze(0), next_logits_dist[sampled_action_idx]
                # convert back to indices of tokenizer
                next_logits = token_mask.nonzero()[:,1][next_logits].unsqueeze(0)
            else:
                # (bs,)
                next_logit_scores = next_logit_score_dist.gather(-1,sampled_action_idx.unsqueeze(-1)).squeeze(-1)
                next_logits = next_logits_dist.gather(-1,sampled_action_idx.unsqueeze(-1)).squeeze(-1)

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
        return {'actions': next_actions, 'action_seq': action_sequence, 'action_seq_mask': action_sequence_mask, 'states_seq': image_sequence, 'log_probs': scores}, entropy

    def score_all_continuations(self, goal_representation, action_seq_past, image_seq_w_curr, i_mask, o_mask, continuations: list, temperature: int=1.0):
        bs = goal_representation.size(0)

        action_sequence, action_sequence_mask, image_sequence, fused_action_image_rep = self.setup_inputs_for_generate(
            goal_representation, action_seq_past, image_seq_w_curr, i_mask=i_mask, o_mask=o_mask,
        )
        # return scores of all possible actions
        encoder_outputs = self.model.encoder(
            input_ids=goal_representation, attention_mask=i_mask,
        )
        # (n_actions, n_tokens)
        all_action_tokens = self.tokenizer(continuations, return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
        n_actions, n_tokens_in_actions = all_action_tokens['input_ids'].size(0), all_action_tokens['input_ids'].size(1)

        bs = goal_representation.size(0)
        batch_action_scores = []
        # # next action embeds
        # # (n_actions, n_tokens_in_actions, word_embed_dim)
        # all_action_embeds = self.model.decoder.embed_tokens(all_action_tokens['input_ids'])
        batch_all_next_actions = []
        batch_all_next_actions_masks = []
        batch_all_next_actions_imgs = []

        next_token_positions = torch.stack([asm.nonzero().max() + 1 for asm in action_sequence_mask])
        # (bs, n_tokens)
        all_next_actions_w_seq = action_sequence.clone()
        all_next_actions_w_seq_mask = action_sequence_mask.clone()
        all_next_action_imgs = image_sequence.clone()
        # add sequence to end
        if action_sequence.size(1) < n_tokens_in_actions + next_token_positions.max():
            # (bs,n_tokens+n_tokens_in_actions)
            all_next_actions_w_seq = F.pad(all_next_actions_w_seq, pad=(0,n_tokens_in_actions+next_token_positions.max()-action_sequence.size(1),0,0), value=self.tokenizer.pad_token_id)
            all_next_actions_w_seq_mask = F.pad(all_next_actions_w_seq_mask, pad=(0,n_tokens_in_actions+next_token_positions.max()-action_sequence_mask.size(1),0,0), value=0)
            # repeat last state
            # (bs, n_tokens+n_tokens_in_actions, image_dim)
            all_next_action_imgs = F.pad(image_sequence, pad=(0,0,0,n_tokens_in_actions+next_token_positions.max()-image_sequence.size(1),0,0), value=0.0)

        for i in range(bs):
            next_token_pos = next_token_positions[i]
            """
            make new action sequence
            """
            # (n_tokens+n_tokens_in_actions) -> (n_actions, n_tokens+n_tokens_in_actions)
            all_next_actions_w_seq_i = all_next_actions_w_seq[i].unsqueeze(0).repeat(n_actions,1)
            all_next_actions_w_seq_mask_i = all_next_actions_w_seq_mask[i].unsqueeze(0).repeat(n_actions,1)
            # (n_actions, n_tokens+n_tokens_in_actions)
            all_next_actions_w_seq_i[:,next_token_pos:next_token_pos+n_tokens_in_actions] = all_action_tokens['input_ids']
            all_next_actions_w_seq_mask_i[:,next_token_pos:next_token_pos+n_tokens_in_actions] = all_action_tokens['attention_mask']
            # [(n_actions, n_tokens+n_tokens_in_actions) x i]
            batch_all_next_actions.append(all_next_actions_w_seq_i)
            batch_all_next_actions_masks.append(all_next_actions_w_seq_mask_i)

            """
            make new image sequence
            """
            # (bs, n_actions, image_dim) -> (n_actions, n_tokens+n_tokens_in_actions, image_dim)
            all_next_action_imgs_i = all_next_action_imgs[i].unsqueeze(0).repeat(n_actions,1,1)
            all_next_action_imgs_i[:,next_token_pos:next_token_pos+n_tokens_in_actions,:] = all_next_action_imgs_i[:,next_token_pos-1,:].unsqueeze(1).repeat(1,n_tokens_in_actions,1)
            all_next_action_imgs_i[~all_next_actions_w_seq_mask_i.bool()] = 0.0  # mask out padding
            batch_all_next_actions_imgs.append(all_next_action_imgs_i)

            # (n_actions, n_tokens_in_actions, word_embed_dim + image_dim) -> (n_actions, n_tokens_in_actions, linear_out_dim)
            all_next_actions_w_seq_embed_i = self.model.decoder.embed_tokens(all_next_actions_w_seq_i)
            fused_next_action_image_rep = self.fusion_module(torch.cat([all_next_actions_w_seq_embed_i, all_next_action_imgs_i], dim=-1))
            next_action_pred = all_next_actions_w_seq_i[:,1:].clone()
            next_action_pred[next_action_pred[:, :] == self.tokenizer.pad_token_id] = -100
            fused_next_action_image_rep_inputs = fused_next_action_image_rep[:,:-1,:]

            model_outputs = self.model(
                encoder_outputs=(encoder_outputs.last_hidden_state[i].unsqueeze(0).repeat(n_actions,1,1),),
                decoder_inputs_embeds=fused_next_action_image_rep_inputs, decoder_attention_mask=all_next_actions_w_seq_mask_i[:,:-1],
                labels=next_action_pred,
            )
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            scaled_logits = model_outputs.logits / temperature
            # (n_actions x n_tokens_in_actions)
            action_scores = -loss_fct(scaled_logits.view(-1, scaled_logits.size(-1)), next_action_pred.view(-1))
            # (n_actions, n_tokens_in_actions) -> (n_actions)
            action_scores = action_scores.view(n_actions,-1).sum(-1)
            batch_action_scores.append(action_scores)
        # (bs, n_actions)
        batch_action_scores = torch.stack(batch_action_scores, dim=0)
        # (bs, n_actions, n_tokens+n_tokens_in_actions)
        batch_all_next_actions = torch.stack(batch_all_next_actions, dim=0)
        # (bs, n_actions, n_tokens+n_tokens_in_actions)
        batch_all_next_actions_masks = torch.stack(batch_all_next_actions_masks, dim=0)
        # (bs, n_actions, n_tokens+n_tokens_in_actions, image_dim)
        batch_all_next_actions_imgs = torch.stack(batch_all_next_actions_imgs, dim=0)
        if bs == 1:
            best_action = batch_action_scores.argmax(-1)[0]
            return batch_action_scores, {'action_seq': batch_all_next_actions[:,best_action], 'action_seq_mask': batch_all_next_actions_masks[:,best_action], 'states_seq': batch_all_next_actions_imgs[:,best_action], 'actions' : all_action_tokens['input_ids'][best_action].unsqueeze(0)}
        else:
            best_action = batch_action_scores.argmax(-1)
            return batch_action_scores, {'action_seq': batch_all_next_actions[torch.arange(bs),best_action], 'action_seq_mask': batch_all_next_actions_masks[torch.arange(bs),best_action], 'states_seq': batch_all_next_actions_imgs[torch.arange(bs),best_action], 'actions' : all_action_tokens['input_ids'][best_action]}

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

    def featurize(self, batch, instr_idx = None):
        """
        if instr_idx is None we use all of instructions and also the goal
        if instr_idx is not None we condition the transformer on
        instructions[instr_idx - 1:]. e.g. if instr_idx=-1 thena we condition on the
        last subgoal and <<stop>>. note we do not condition on the goal itself.
        """
        feat = {
            "goal_representation": [],
        }
        for item in batch:
            instructions = item['ann']['instr']
            if instr_idx is None:
                feat['goal_representation'].append(
                    ''.join([g.rstrip() for g in item['ann']['goal']]).replace('<<goal>>', ' [goal]').replace('<<stop>>', ' [stop]').replace('  ', ' ').strip()
                )
            else:
                feat['goal_representation'].append('')
                instructions = [item['ann']['instr'][instr_idx - 1]]
            for instr in instructions:
                feat['goal_representation'][-1] += (
                    ''.join([i.rstrip() for i in instr])
                ).replace('<<goal>>', ' [goal]').replace('<<stop>>', ' [stop]').replace('  ', ' ').strip()
        print("goal representation: ", feat['goal_representation'])
        feat['goal_representation'] = self.tokenizer(feat['goal_representation'], return_tensors='pt', padding=True).to(self.device)
        feat['actions'] = {
            'input_ids': torch.Tensor(len(batch),0).long().to(self.device),
            'attention_mask': torch.Tensor(len(batch),0).long().to(self.device),
        }
        return feat

    @classmethod
    def generate_action_mask(cls, action, curr_image, image_obj_features):
        m = np.zeros((300, 300))
        # extract object(s) from action
        object_names = action.split(':')[-1].strip()
        object_names = [object_names.split(' in ')[-1]] # only want the destination object
        if len(object_names) == 0: return None
        # ["table", "chair"]
        obj_label2feature_idx = dict()
        for i, label in enumerate(image_obj_features["class_labels"]):
            if CLASSES[label] not in obj_label2feature_idx:
                obj_label2feature_idx[CLASSES[label]] = i # TODO consider more aggressive probability filtering
        obj_masks = []
        for obj_name in object_names:
            # TODO(sahit): switch from t/e to defaultdict
            try:
                obj_idx = obj_label2feature_idx[snake_to_camel(obj_name)]
            except Exception as e:
                print(f"trying to interact with: {obj_name} not in the scene")
                obj_name = ' ' if obj_name == '' else obj_name
                list_of_objs = image_obj_features['class_labels']
                distances = [snake_to_camel(obj_name) in CLASSES[scene_elem] for scene_elem in list_of_objs]
                distances = [-1*int(x) for x in distances]
                minimizing_idx = np.argmin(distances)
                obj_name_similar = CLASSES[list_of_objs[minimizing_idx]]
                print(f"replacing with {obj_name_similar}")
                obj_idx = minimizing_idx
            obj_masks.append(image_obj_features['masks'][obj_idx][0])
        m = obj_masks[0]
        return m

    def decode_prediction(self, m_out, curr_image, object_features):
        # TODO: Maybe cast all actions to be within the 13 valid tokens.
        # Also this is a good place to add in the exploration
        action = self.tokenizer.decode(m_out, skip_special_tokens=True).split(",")[0].strip()
        if action == "stop>>":
            action = "<<stop>>"
        if action.startswith("[subgoal]"):
            action = "[subgoal]"
        # convert BACK to API action!!!
        api_action = snake_to_camel(action.split(':')[0].strip())
        if api_action not in API_ACTIONS:
            breakpoint()
            api_action =  API_ACTIONS[4]
        assert api_action in API_ACTIONS, f"{action} is not part of {API_ACTIONS}!"
        mask = (
            self.generate_action_mask(action, curr_image, object_features)
            if self.has_interaction(api_action)
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
        # TODO(sahit): go back to O.G has_interaction? messing w this fn is wrong
        return action in ["PickupObject", "ToggleObject", "ToggleObjectOn", "ToggleObjectOff",
                "PutObject", "SliceObject", "OpenObject", "CloseObject"]
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
