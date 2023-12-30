import argparse
import sys
import traceback
from gymnasium.spaces import Discrete

import numpy as np
import torch



sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from policy_offline_utils import prepare_buffer_via_offline_data
from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, prepare_test_envs, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.util.data import get_env_args
from core.policy.RecPolicy import RecPolicy

from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import DiscreteCRRPolicy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_CRR():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DiscreteCRR")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--message", type=str, default="DiscreteCRR")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, buffer, test_envs_dict):

    net = Net(args.state_dim, args.hidden_sizes[0], device=args.device)
    actor = Actor(
        net,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # softmax_output=False  # 如果不经过softmax, dist概率采样会有bug
    )
    critic = Critic(
        net,
        hidden_sizes=args.hidden_sizes,
        last_size=np.prod(args.action_shape),
        device=args.device
    )
    actor_critic = ActorCritic(actor, critic)
    optim_RL = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    policy = DiscreteCRRPolicy(
        actor,
        critic,
        optim,
        args.gamma,
        target_update_freq=args.target_update_freq,
        state_tracker=state_tracker,
        buffer=buffer,
        action_space=Discrete(args.action_shape),
    ).to(args.device)

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
    env, dataset, kwargs_um, buffer = prepare_buffer_via_offline_data(args)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, buffer, test_envs_dict, use_buffer_in_train=True)
    policy, test_collector_set, optim = setup_policy_model(args, state_tracker, buffer, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path, trainer="offline")


if __name__ == '__main__':
    trainer = "offline"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_CRR = get_args_CRR()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_CRR.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
