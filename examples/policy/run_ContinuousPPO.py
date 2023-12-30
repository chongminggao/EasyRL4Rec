import argparse
import sys
import traceback
from gymnasium.spaces import Box
from torch.distributions import Independent, Normal

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, setup_state_tracker, \
    prepare_train_test_envs

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.util.data import get_env_args
from core.collector.collector import Collector
from core.policy.RecPolicy import RecPolicy

from tianshou.data import VectorReplayBuffer

from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import PPOPolicy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_PPO():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ContinuousPPO")
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--norm-adv', action="store_true", default=True)
    parser.add_argument('--recompute-adv', action="store_true", default=False)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', action="store_true", default=True)
    # parser.add_argument('--resume', action="store_true")

    parser.add_argument("--message", type=str, default="ContinuousPPO")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)

    # 连续动作，不能在用设置state_tracker时设置的action_shape todo
    actor = ActorProb(net, state_tracker.emb_dim, unbounded=True,
                      device=args.device).to(args.device)
    critic = Critic(
        Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device),
        device=args.device
    ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # if torch.cuda.is_available():
    #     actor = DataParallelNet(
    #         Actor(net, args.action_shape, device=None).to(args.device)
    #     )
    #     critic = DataParallelNet(Critic(net, device=None).to(args.device))
    # else:
    #     actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    #     critic = Critic(net, device=args.device).to(args.device)

    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    optim_RL = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=Box(shape=(state_tracker.emb_dim,), low=0, high=1),
        # deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        action_bound_method="",  # not clip  # follow A2C?
        action_scaling=False
    )

    rec_policy = RecPolicy(args, policy, state_tracker)

    # Prepare the collectors and logs
    train_collector = Collector(
        rec_policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        # preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
        remove_recommended_ids=args.remove_recommended_ids
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
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs,
                                                                            test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="onpolicy")


if __name__ == '__main__':
    trainer = "onpolicy"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_PPO = get_args_PPO()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_PPO.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
