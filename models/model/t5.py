import os
import torch
import json
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer

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

    def train_forward(self, goal_representation, action_sequence, image_sequence,i_mask,o_mask):
        # encode goal with T5 encoder
        # assume action_sequence doesn't have start id
        action_sequence_shifted = action_sequence.new_zeros(action_sequence.shape)
        # action_sequence_shifted[..., 1:] = action_sequence_shifted[..., :-1].clone()
        action_sequence_shifted[..., 1:] = action_sequence[..., :-1].clone()
        action_sequence_shifted[..., 0] = self.model.config.decoder_start_token_id

        # pad `image_sequence`
        image_sequence_padded = torch.cat([image_sequence[:,0,...].unsqueeze(1), image_sequence], dim=1).view(*action_sequence.size(), -1)
        embedded_action_sequence = self.model.decoder.embed_tokens(action_sequence_shifted)
        fused_action_image_rep = self.fusion_module(torch.cat([embedded_action_sequence, image_sequence_padded], dim=-1))

        action_sequence[action_sequence[:, :] == self.tokenizer.pad_token_id] = -100
        return self.model(input_ids=goal_representation,attention_mask=i_mask,
                          decoder_inputs_embeds=fused_action_image_rep,
                          decoder_attention_mask=o_mask,
                          labels = action_sequence)

    def test_generate(self, goal_representation, action_sequence, image_sequence,i_mask,o_mask):
        # encode goal with T5 encoder
        # assume action_sequence doesn't have start id
        action_sequence_shifted = action_sequence.new_zeros(action_sequence.shape)
        # action_sequence_shifted[..., 1:] = action_sequence_shifted[..., :-1].clone()
        action_sequence_shifted[..., 1:] = action_sequence[..., :-1].clone()
        action_sequence_shifted[..., 0] = self.model.config.decoder_start_token_id

        # pad `image_sequence`
        image_sequence_padded = torch.cat([image_sequence[:,0,...].unsqueeze(1), image_sequence], dim=1).view(*action_sequence.size(), -1)
        embedded_action_sequence = self.model.decoder.embed_tokens(action_sequence_shifted)
        fused_action_image_rep = self.fusion_module(torch.cat([embedded_action_sequence, image_sequence_padded], dim=-1))

        action_sequence[action_sequence[:, :] == self.tokenizer.pad_token_id] = -100
        return self.model.generate(input_ids=goal_representation,
                          decoder_inputs_embeds=fused_action_image_rep,max_length=300,do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)

    @classmethod
    def load(cls, args, fsave):
        model = cls(args=args)
        model.load_state_dict(torch.load(fsave))
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
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = {
            'all_states': [],
            'all_actions': [],
            'all_actions_mask': [],
            "goal_representation": [],
        }
        for item in batch:
            # dict_keys(['ann', 'images', 'num', 'pddl_params', 'plan', 'repeat_idx', 'root', 'scene', 'split', 'task_id', 'task_type', 'turk_annotations'])
            feat['goal_representation'] = ' '.join(item['ann']['goal'])
            # TODO: trim <<goal>> and <<stop>> from goal rep?
            for instr in item['ann']['instr']:
                feat['goal_representation'] += (' '.join(instr))
            feat['goal_representation'] = self.tokenizer.encode(feat['goal_representation'])
            feat['goal_representation'] = torch.tensor([feat['goal_representation']], dtype=torch.int).to(device)

        return feat

