import argparse
import sys
import traceback
from gymnasium.spaces import Box

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, setup_state_tracker, \
    prepare_train_test_envs

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.core.collector.collector_set import CollectorSet
from src.core.util.data import get_env_args
from src.core.collector.collector import Collector
from src.core.policy.RecPolicy import RecPolicy

from src.tianshou.tianshou.data import VectorReplayBuffer

from src.tianshou.tianshou.utils.net.common import Net
from src.tianshou.tianshou.utils.net.continuous import Actor, Critic
from src.tianshou.tianshou.policy import DDPGPolicy
from src.tianshou.tianshou.exploration import GaussianNoise

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_DDPG():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DDPG")
    # ddpg special
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)

    parser.add_argument('--remap_eps', default=0.01, type=float)
    parser.add_argument('--rew-norm', action="store_true", default=False)

    parser.add_argument("--message", type=str, default="DDPG")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # Continuous Action，action_shape = state_tracker.emb_dim
    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(
        net, state_tracker.emb_dim, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net = Net(
        args.state_dim,
        state_tracker.emb_dim,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic = Critic(net, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)

    policy = DDPGPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        optim_state,
        state_tracker=state_tracker,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.explore_eps),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=Box(shape=(state_tracker.emb_dim,), low=0, high=1),
    )

    rec_policy = RecPolicy(args, policy, state_tracker)

    # Prepare the collectors and logs
    assert args.exploration_noise == True
    train_collector = Collector(
        rec_policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=args.exploration_noise,
        remove_recommended_ids=args.remove_recommended_ids
    )

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                      #   preprocess_fn=state_tracker.build_state,
                                      #   exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return rec_policy, train_collector, test_collector_set, [actor_optim, optim_state]  # TODO


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


if __name__ == '__main__':
    trainer = "offpolicy"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_DDPG = get_args_DDPG()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_DDPG.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
