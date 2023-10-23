import argparse
import functools
import os
import pprint
import sys
import traceback
from gymnasium.spaces import Discrete

import torch

sys.path.extend([".", "./examples", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy.policy_utils import get_args_all, prepare_dir_log, prepare_user_model, prepare_buffer_via_offline_data, prepare_test_envs, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.evaluation.evaluator import Evaluator_Feat, Evaluator_Coverage_Count, Evaluator_User_Experience, save_model_fn
from core.evaluation.loggers import LoggerEval_Policy
from core.util.layers import Actor_Linear
from core.util.data import get_env_args
from core.policy.RecPolicy import RecPolicy

from tianshou.utils.net.common import ActorCritic
from tianshou.policy import SQNPolicy
from tianshou.trainer import offline_trainer

# from util.upload import my_upload
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_SQN():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SQN")
    parser.add_argument('--which_head', type=str, default='qhead')  # in {"shead", "qhead", "bcq"}

    # bcq
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.6)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)
    parser.add_argument("--eps-test", type=float, default=0.001)
    # parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)

    # parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="SQN")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, buffer, test_envs_dict):

    model_final_layer = Actor_Linear(args.state_dim, args.action_shape, device=args.device).to(args.device)
    imitation_final_layer = Actor_Linear(args.state_dim, args.action_shape, device=args.device).to(args.device)

    actor_critic = ActorCritic(model_final_layer, imitation_final_layer)
    optim_RL = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    policy = SQNPolicy(
        model_final_layer,
        imitation_final_layer,
        optim,
        args.gamma,
        args.n_step,
        args.target_update_freq,
        args.eps_test,
        args.unlikely_action_threshold,
        args.imitation_logits_penalty,
        state_tracker=state_tracker,
        buffer=buffer,
        which_head=args.which_head,
        action_space=Discrete(args.action_shape),
    )
    policy.set_eps(args.eps_test)

    rec_policy = RecPolicy(args, policy, state_tracker)

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                    #   preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return rec_policy, test_collector_set, optim


def learn_policy(args, env, dataset, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path):
    # log
    # t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # log_file = f'seed_{args.seed}_{t0}-{args.env.replace("-", "_")}_bcq'
    # log_path = os.path.join(args.logdir, args.env, 'bcq', log_file)
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(args))
    # logger1 = TensorboardLogger(writer)
    #
    # def save_best_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    df_val, df_user_val, df_item_val, list_feat = dataset.get_val_data()
    item_feat_domination = dataset.get_domination()
    item_similarity, item_popularity = dataset.get_item_similarity(), dataset.get_item_popularity()

    # set metrics and related evaluator
    metrics = ['len_tra', 'R_tra', 'ctr', 'CV', 'CV_turn', 'ifeat_', 'Diversity', 'Novelty']

    policy.callbacks = [
        Evaluator_Feat(test_collector_set, df_item_val, args.need_transform, item_feat_domination,
                                lbe_item=env.lbe_item if args.need_transform else None, top_rate=args.top_rate, draw_bar=args.draw_bar),
        Evaluator_Coverage_Count(test_collector_set, df_item_val, args.need_transform),
        Evaluator_User_Experience(test_collector_set, df_item_val, item_similarity, item_popularity,
                                  args.need_transform, lbe_item=env.lbe_item if args.need_transform else None),
        LoggerEval_Policy(args.force_length, metrics)]

    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    result = offline_trainer(
        policy,
        buffer,
        test_collector_set,
        args.epoch,
        args.step_per_epoch,
        args.test_num,
        args.batch_size,
        # save_best_fn=save_best_fn,
        # stop_fn=stop_fn,
        # logger=logger1,
        save_model_fn=functools.partial(save_model_fn,
                                        model_save_path=model_save_path,
                                        state_tracker=state_tracker,
                                        optim=optim,
                                        is_save=args.is_save)
    )

    print(__file__)
    pprint.pprint(result)
    logger.info(result)


def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, dataset, buffer = prepare_buffer_via_offline_data(args)
    test_envs_dict = prepare_test_envs(args)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, buffer, test_envs_dict, use_buffer_in_train=True)
    policy, test_collector_set, optim = setup_policy_model(args, state_tracker, buffer, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_env_args(args_all)
    args_SQN = get_args_SQN()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_SQN.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
