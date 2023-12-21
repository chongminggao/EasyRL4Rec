import argparse
import functools
import os
import pprint
import sys
import traceback
from gymnasium.spaces import Box

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, prepare_buffer_via_offline_data, prepare_test_envs, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.evaluation.evaluator import Evaluator_Feat, Evaluator_Coverage_Count, Evaluator_User_Experience, save_model_fn
from core.evaluation.loggers import LoggerEval_Policy
from core.util.data import get_env_args
from core.policy.RecPolicy import RecPolicy

from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import VAE, Critic, Perturbation
from tianshou.policy import BCQPolicy
from tianshou.trainer import offline_trainer

# from util.upload import my_upload
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_BCQ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ContinuousBCQ")
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64])
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    
    parser.add_argument("--n-step", type=int, default=3)
    # parser.add_argument("--update-per-epoch", type=int, default=5000)

    parser.add_argument("--vae-hidden-sizes", type=int, nargs='*', default=[32, 32])
    # default to 2 * action_dim
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--tau", default=0.005)
    # Weighting for Clipped Double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)
    # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--phi", default=0.05)

    
    parser.add_argument("--message", type=str, default="ContinuousBCQ")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, buffer, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    args.action_dim = state_tracker.emb_dim
    args.max_action = 1.
    print("args.action_dim", args.action_dim)
    # model
    # perturbation network
    net_a = MLP(
        input_dim=args.state_dim + args.action_dim,
        output_dim=args.action_dim,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = Perturbation(
        net_a, max_action=args.max_action, device=args.device, phi=args.phi
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(
        args.state_dim,
        args.action_dim,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_dim,
        args.action_dim,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # vae
    # output_dim = 0, so the last Module in the encoder is ReLU
    vae_encoder = MLP(
        input_dim=args.state_dim + args.action_dim,
        hidden_sizes=args.vae_hidden_sizes,
        device=args.device,
    )
    if not args.latent_dim:
        args.latent_dim = args.action_dim * 2
    vae_decoder = MLP(
        input_dim=args.state_dim + args.latent_dim,
        output_dim=args.action_dim,
        hidden_sizes=args.vae_hidden_sizes,
        device=args.device,
    )
    vae = VAE(
        vae_encoder,
        vae_decoder,
        hidden_dim=args.vae_hidden_sizes[-1],
        latent_dim=args.latent_dim,
        max_action=args.max_action,
        device=args.device,
    ).to(args.device)
    vae_optim = torch.optim.Adam(vae.parameters())

    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [actor_optim, optim_state]

    policy = BCQPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        vae,
        vae_optim,
        optim_state,
        device=args.device,
        gamma=args.gamma,
        tau=args.tau,
        lmbda=args.lmbda,
        state_tracker=state_tracker,
        buffer=buffer,
        action_space=Box(shape=(state_tracker.emb_dim,), low=0, high=1),
    )

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
    learn_policy(args, env, dataset, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path, trainer="offline")


if __name__ == '__main__':
    args_all = get_args_all()
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
