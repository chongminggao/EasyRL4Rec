import argparse
import numpy as np
import sys
import traceback
from gymnasium.spaces import Box
from torch.distributions import Independent, Normal

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, setup_state_tracker, prepare_train_test_envs

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.util.data import get_env_args
from core.collector.collector import Collector
from core.policy.RecPolicy import RecPolicy


from tianshou.data import VectorReplayBuffer

from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb
from tianshou.policy import PGPolicy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_PG():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ContinuousPG")
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--action-scaling', action="store_true", default=True)
    parser.add_argument('--action-bound-method', type=str, default="clip")

    
    parser.add_argument("--message", type=str, default="ContinuousPG")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device).to(args.device)
    actor = ActorProb(net, state_tracker.emb_dim, unbounded=True,
                      device=args.device).to(args.device)
    optim_RL = torch.optim.Adam(actor.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    for m in net.modules():  # follow test_py.py
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    policy = PGPolicy(
        actor,
        optim,
        dist,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        reward_normalization=args.rew_norm,
        action_space=Box(shape=(state_tracker.emb_dim,), low=0, high=1),
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,  # clip by default
    )

    rec_policy = RecPolicy(args, policy, state_tracker)

    # Prepare the collectors and logs
    train_collector = Collector(
        rec_policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        # preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
        remove_recommended_ids = args.remove_recommended_ids
    )

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
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="onpolicy")


if __name__ == '__main__':
    trainer = "onpolicy"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_PG = get_args_PG()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_PG.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
