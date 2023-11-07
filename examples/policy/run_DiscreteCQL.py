import argparse
import functools
import os
import pprint
import sys
import traceback
from gymnasium.spaces import Discrete

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, prepare_buffer_via_offline_data, prepare_test_envs, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.evaluation.evaluator import Evaluator_Feat, Evaluator_Coverage_Count, Evaluator_User_Experience, save_model_fn
from core.evaluation.loggers import LoggerEval_Policy
from core.util.data import get_env_args
from core.policy.RecPolicy import RecPolicy

from tianshou.utils.net.common import Net
from tianshou.policy import DiscreteCQLPolicy
from tianshou.trainer import offline_trainer

# from util.upload import my_upload\
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_CQL():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DiscreteCQL")
    parser.add_argument('--num-quantiles', type=int, default=20)
    parser.add_argument("--min-q-weight", type=float, default=10.)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument("--message", type=str, default="DiscreteCQL")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, buffer, test_envs_dict):

    net = Net(
        args.state_dim,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=False,
        num_atoms=args.num_quantiles
    )
    optim_RL = torch.optim.Adam(net.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    policy = DiscreteCQLPolicy(
        net,
        optim,
        args.gamma,
        args.num_quantiles,
        args.n_step,
        args.target_update_freq,
        min_q_weight=args.min_q_weight,
        state_tracker=state_tracker,
        buffer=buffer,
        action_space=Discrete(args.action_shape),
    ).to(args.device)
    policy.set_eps(args.eps_test)

    rec_policy = RecPolicy(args, policy, state_tracker)

    # collector
    # buffer has been gathered

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                    #   preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return rec_policy, test_collector_set, optim



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
    learn_policy(args, env, dataset, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path, is_offline=True)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_env_args(args_all)
    args_CQL = get_args_CQL()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_CQL.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
