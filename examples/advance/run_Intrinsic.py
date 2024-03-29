import argparse
import sys
import traceback
import pickle
import random

import numpy as np
import torch
from gymnasium.spaces import Discrete

sys.path.extend([".", "./examples", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from ..policy.policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, prepare_test_envs, \
    setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.core.collector.collector_set import CollectorSet
from src.core.util.data import get_env_args, get_true_env
from src.core.collector.collector import Collector
from src.core.envs.Simulated_Env.intrinsic import IntrinsicSimulatedEnv
from src.core.policy.RecPolicy import RecPolicy

from src.tianshou.tianshou.data import VectorReplayBuffer
from src.tianshou.tianshou.env import DummyVectorEnv
from src.tianshou.tianshou.utils.net.common import ActorCritic, Net
from src.tianshou.tianshou.utils.net.discrete import Actor, Critic
from src.tianshou.tianshou.policy import A2CPolicy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_Intrinsic():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Intrinsic")
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)

    # Env
    parser.add_argument('--lambda_diversity', default=0.1, type=float)
    parser.add_argument('--lambda_novelty', default=0.1, type=float)
    
    parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="Intrinsic")

    args = parser.parse_known_args()[0]
    return args


def prepare_train_envs_local(args, ensemble_models, env, dataset, kwargs_um):
    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    item_similarity, item_popularity = dataset.get_item_similarity(), dataset.get_item_popularity()

    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,

        "item_similarity": item_similarity,
        "item_popularity": item_popularity,
        "lambda_diversity": args.lambda_diversity,
        "lambda_novelty": args.lambda_novelty,
    }

    train_envs = DummyVectorEnv(
        [lambda: IntrinsicSimulatedEnv(**kwargs) for _ in range(args.training_num)])
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)

    return train_envs


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    optim_RL = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor,
        critic,
        optim,
        dist,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=Discrete(args.action_shape),
        action_bound_method="",  # not clip
        action_scaling=False
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
    env, dataset, kwargs_um = get_true_env(args)
    train_envs = prepare_train_envs_local(args, ensemble_models, env, dataset, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)

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
    args_Intrinsic = get_args_Intrinsic()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_Intrinsic.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
