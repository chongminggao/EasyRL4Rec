import argparse
import sys
import traceback
from gymnasium.spaces import Discrete

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from policy_offline_utils import prepare_buffer_via_offline_data, get_args_offline
from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, prepare_test_envs, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.util.data import get_env_args
from core.policy.RecPolicy import RecPolicy

from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor
from tianshou.policy import DiscreteBCQPolicy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_BCQ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DiscreteBCQ")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.6)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)
    # parser.add_argument("--update-per-epoch", type=int, default=5000)
    
    parser.add_argument("--message", type=str, default="DiscreteBCQ")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, buffer, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(args.state_dim, args.hidden_sizes[0], device=args.device)
    policy_net = Actor(
        net, args.action_shape, hidden_sizes=args.hidden_sizes, device=args.device
    ).to(args.device)
    imitation_net = Actor(
        net, args.action_shape, hidden_sizes=args.hidden_sizes, device=args.device
    ).to(args.device)
    actor_critic = ActorCritic(policy_net, imitation_net)
    optim_RL = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    policy = DiscreteBCQPolicy(
        policy_net,
        imitation_net,
        optim,
        args.gamma,
        args.n_step,
        args.target_update_freq,
        args.explore_eps,
        args.unlikely_action_threshold,
        args.imitation_logits_penalty,
        state_tracker=state_tracker,
        buffer=buffer,
        action_space=Discrete(args.action_shape),
    )
    policy.set_eps(args.explore_eps)

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
    args_all = get_args_offline(args_all)
    args = get_env_args(args_all)
    args_BCQ = get_args_BCQ()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_BCQ.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
