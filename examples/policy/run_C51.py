import argparse
import functools
import os
import pprint
import sys
import traceback
from gymnasium.spaces import Discrete

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_test_envs, prepare_train_envs, prepare_user_model, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.util.data import get_env_args
from core.collector.collector import Collector
from core.policy.RecPolicy import RecPolicy


from tianshou.data import VectorReplayBuffer, PrioritizedVectorReplayBuffer

from tianshou.utils.net.common import Net
from tianshou.policy import C51Policy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_C51():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="C51")

    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--reward-normalization', action="store_true", default=False)
    parser.add_argument('--is-double', type=bool, default=True)
    parser.add_argument('--clip-loss-grad', action="store_true", default=False)
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.)
    parser.add_argument('--v-max', type=float, default=10.)
    
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--training-num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--is_random_init', dest='random_init', action='store_true')
    parser.add_argument('--no_random_init', dest='random_init', action='store_false')
    parser.set_defaults(random_init=True)

    parser.add_argument('--is_exploration_noise', dest='exploration_noise', action='store_true')
    parser.add_argument('--no_exploration_noise', dest='exploration_noise', action='store_false')
    parser.set_defaults(exploration_noise=True)
    parser.add_argument('--eps', default=0.2, type=float)

    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--prioritized-replay', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    # parser.add_argument('--resume', action="store_true")  # yyq:暂时不考虑load from existing checkpoint
    # parser.add_argument("--save-interval", type=int, default=4)

    parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="C51")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(
        args.state_dim,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=True,
        num_atoms=args.num_atoms,
    )
    optim_RL = torch.optim.Adam(net.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]
    policy = C51Policy(
        net,
        optim,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        reward_normalization=args.reward_normalization,
        is_double=args.is_double, 
        clip_loss_grad=args.clip_loss_grad, 
        action_space=Discrete(args.action_shape),
    ).to(args.device)
    policy.set_eps(args.eps)  ## args.eps_test

    rec_policy = RecPolicy(args, policy, state_tracker)

    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    # Prepare the collectors and logs
    train_collector = Collector(
        rec_policy, train_envs,
        buffer=buf,
        # preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
    )

    rec_policy.set_collector(train_collector)  ## TODO
    # train_collector.collect(n_step=args.batch_size * args.training_num)  ## TODO

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                    #   preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return rec_policy, train_collector, test_collector_set, optim


def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, dataset, train_envs = prepare_train_envs(args, ensemble_models)
    test_envs_dict = prepare_test_envs(args)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, is_onpolicy=False)


if __name__ == "__main__":
    args_all = get_args_all()
    args = get_env_args(args_all)
    args_C51 = get_args_C51()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_C51.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
