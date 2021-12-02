import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import argparse
import torch.multiprocessing as mp
from eval_task import EvalTask
from eval_subgoals import EvalSubgoals


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--data', type=str, default="data/json_2.1.0")
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['train', 'valid_seen', 'valid_unseen'])
    parser.add_argument('--model_path', type=str, default="model.pth")
    parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--use_templated_goals', help='use templated goals instead of human-annotated goal descriptions (only available for train set)', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')

    # eval params
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')

    # eval settings
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--force_last_subgoal', dest='force_last_subgoal', action='store_true')
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--skip_model_unroll_with_expert', action='store_true', help='forward model with expert actions')
    parser.add_argument('--no_teacher_force_unroll_with_expert', action='store_true', help='no teacher forcing with expert')

    # debug
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--fast_epoch', dest='fast_epoch', action='store_true')

    ## MaskRCNN parameters
    parser.add_argument('--box_score_thresh', type=float, default=0.05)
    parser.add_argument('--box_nms_thresh', type=float, default=0.5)
    parser.add_argument('--panoramic_boxes', nargs="+", default=(36, 18, 18, 18), type=int)
    parser.add_argument('--max_boxes_per_image', type=int, default=36)
    parser.add_argument('--frame_size', type=int, default=300)
    parser.add_argument('--maskrcnn_checkpoint', default="storage/models/vision/moca_maskrcnn/weight_maskrcnn.pt",
                        type=str)
    # parse arguments
    args = parser.parse_args()

    # eval mode
    if args.subgoals:
        eval = EvalSubgoals(args, manager)
    else:
        eval = EvalTask(args, manager)

    # start threads
    eval.spawn_threads()
