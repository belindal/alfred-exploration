import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from datetime import datetime
from models.eval.eval import Eval
from env.thor_env import ThorEnv
from scripts.generate_maskrcnn import MaskRCNNDetector, CustomImageLoader
from models.model.t5 import vis_encoder, API_ACTIONS, CLASSES, API_ACTIONS_NATURALIZED, unCamelSnakeCase, API_ACTIONS_SN
from models.utils.debug_utils import plot_mask
import gen.constants
import random
import scripts.exploration_strategies as exploration_strategies

class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, image_loader, region_detector, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()
        runs = 0
        while True:
            runs += 1
            if task_queue.qsize() == 0:
                break
            task = task_queue.get()
            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, image_loader, region_detector, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))
                sys.exit()

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, image_loader, region_detector, traj_data, args, lock, successes, failures, results):
        # reset model
        #model.reset()
        device = torch.device("cpu")
        instr_idx = -args.force_last_k_subgoals
        if vars(args)["gpu"]:
            model = model.to('cuda')
            device = torch.device("cuda")

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # get expert actions for everything but the last subgoal
        num_subgoals = len(traj_data['turk_annotations']['anns'][r_idx]['high_descs'])
        # (i wonder if just one subgoal is too easy?)
        expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] <= num_subgoals - 1 - args.force_last_k_subgoals]
        gt_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions']]


        # extract language features
        feat = model.featurize([traj_data], instr_idx=instr_idx)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        done, success = False, False
        fails = 0
        t = 0
        reward = 0

        state_history = torch.tensor([])
        n = 0
        exploration_strat = exploration_strategies.get_random_exploration_sequence()
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            curr_state = resnet.featurize([curr_image], batch=1)
            curr_state = vis_encoder(curr_state.cpu()).unsqueeze(0)

            object_features = cls.get_visual_features(env, image_loader, region_detector, args, device)
            seen_objects = [unCamelSnakeCase(CLASSES[idx]) for idx in object_features[0]["class_labels"]]

            # forward model
            if t < len(expert_init_actions) and args.force_last_k_subgoals > 0:
                action_dict = expert_init_actions[t]
                action = action_dict['action']
                compressed_mask = action_dict['args']['mask'] if 'mask' in action_dict['args'] else None
                mask = env.decompress_mask(compressed_mask) if compressed_mask is not None else None
            else:
                feat['all_states'] = torch.cat([state_history.cpu(), curr_state], dim=1)
                scores, m_out = model.score_all_continuations(
                    feat["goal_representation"]["input_ids"].to(device),
                    feat['actions']['input_ids'].to(device),
                    feat["all_states"].to(device),
                    i_mask=feat["goal_representation"]["attention_mask"].to(device),
                    o_mask=feat['actions']['attention_mask'].to(device),
                    continuations = [x + ":" for x in API_ACTIONS_SN[:8]] + [x + "," for x in API_ACTIONS_SN[8:]]
                )
                if scores.argmax() < 8:
                    m_out = model.test_generate(
                        feat["goal_representation"]["input_ids"].to(device),
                        feat['actions']['input_ids'].to(device),
                        feat["all_states"].to(device),
                        i_mask=feat["goal_representation"]["attention_mask"].to(device),
                        o_mask=feat['actions']['attention_mask'].to(device),
                        object_list = API_ACTIONS_NATURALIZED + [",", ":", "in"] + seen_objects
                    )
                feat['actions']['input_ids'] = m_out['action_seq']
                feat['actions']['attention_mask'] = m_out['action_seq_mask']
                state_history = m_out['states_seq']
                m_pred = model.decode_prediction(m_out['actions'][0], curr_image, object_features[0])
                #print(m_pred)
                # check if <<stop>> was predicted
                if m_pred['action_low'] == cls.STOP_TOKEN or m_pred['action_low'] == cls.NEW_STOP_TOKEN:
                    print("\tpredicted STOP")
                    break
                elif m_pred['action_low'] == cls.SUBGOAL_STOP_TOKEN:
                    print("\tpredicted [subgoal]")
                    instr_idx += 1
                    feat = model.featurize([traj_data], instr_idx=instr_idx)
                    continue

                # get action and mask
                action, mask = m_pred['action_low'], m_pred['action_low_mask']
                mask = mask if model.has_interaction(action) else None
                if random.random() < 0.0:
                    #action = random.choice(API_ACTIONS)
                    action = random.choice(["RotateRight_90", "RotateLeft_90"])
                    if "_" in action:
                        mask = None
                    else:
                        mask = object_features[0]["masks"][0][0]
                print("nn action: ", action)


            #if args.debug:
            #    plot_mask(feature["masks"][0][0], 'mask.png')

            """
            if n < len(exploration_strat):
                action = exploration_strat[n]
                n+=1
            """
            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            if not t_success:
                print("err: ", err)
                #breakpoint()
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break
            else:
                fails = 0

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True


        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward)}
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

