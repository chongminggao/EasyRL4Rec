import argparse
import sys
import traceback
from gymnasium.spaces import Discrete

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, \
    prepare_user_model, setup_state_tracker, prepare_train_test_envs

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.core.collector.collector_set import CollectorSet
from src.core.util.data import get_env_args
from src.core.collector.collector import Collector
from src.core.policy.RecPolicy import RecPolicy

from src.tianshou.tianshou.data import VectorReplayBuffer, PrioritizedVectorReplayBuffer

from src.tianshou.tianshou.utils.net.common import Net
from src.tianshou.tianshou.policy import DQNPolicy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_DQN():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DQN")

    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--reward-normalization', action="store_true", default=False)
    parser.add_argument('--is-double', type=bool, default=True)
    parser.add_argument('--clip-loss-grad', action="store_true", default=False)

    parser.add_argument('--prioritized-replay', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)

    parser.add_argument("--message", type=str, default="DQN")

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
        # dueling=(Q_param, V_param),
    ).to(args.device)
    optim_RL = torch.optim.Adam(net.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]
    policy = DQNPolicy(
        net,
        optim,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        reward_normalization=args.reward_normalization,
        is_double=args.is_double,
        clip_loss_grad=args.clip_loss_grad,
        action_space=Discrete(args.action_shape),
    )
    policy.set_eps(args.explore_eps)

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
        remove_recommended_ids=args.remove_recommended_ids
    )
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
    env, dataset, train_envs, test_envs_dict = prepare_train_test_envs(args, ensemble_models)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs,
                                                                            test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="offpolicy")


if __name__ == "__main__":
    trainer = "offpolicy"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_DQN = get_args_DQN()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_DQN.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
