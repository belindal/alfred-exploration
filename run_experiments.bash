#!/bin/bash
python3 scripts/parse_results.py new

for i in {1..4}
do
    # (baseline, greedy)
    python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_new_ep0_step80000.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals $i --max_steps 100 --max_fails 5 
    # (baseline, sampling)
    python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_new_ep0_step80000.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals $i --max_steps 100 --max_fails 5 --topk 4
    # (baseline, sampling w temp scaling)
    python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_new_ep0_step80000.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals $i --max_steps 100 --max_fails 5 --topk 4 --decode_temperature 1.5

    # (ls, greedy)
    python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_ls0.2.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals $i --max_steps 100 --max_fails 5 
    # (ls, sampling)
    python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_ls0.2.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals $i --max_steps 100 --max_fails 5 --topk 4
    # (ls, sampling w temp scaling)
    python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_ls0.2.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals $i --max_steps 100 --max_fails 5 --topk 4 --decode_temperature 1.5
done

# the exploration sequence stuff is most likely to OOM so we do it at the end
for i in {1..4}
do
    python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_new_ep0_step80000.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals $i --max_steps 100 --max_fails 5 --naive_explore
    
    python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_ls0.2.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals $i --max_steps 100 --max_fails 5 --naive_explore 
done
