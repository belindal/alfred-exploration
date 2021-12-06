python3 scripts/parse_results.py new
python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_new_ep0_step80000.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals 2 --max_steps 100 --max_fails 5 
python3 models/eval/eval_seq2seq.py --model_path /home/sahit/alfred-exploration/models/subgoals/transformer_ls0.2.pth --eval_split valid_unseen --data /home/sahit/alfred-exploration/data/json_feat_2.1.0 --model models.model.t5 --gpu --num_threads 1 --force_last_k_subgoals 2 --max_steps 100 --max_fails 5 


