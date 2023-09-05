import argparse
import functools
import os
import pprint
import sys
import traceback

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, prepare_dir_log, prepare_user_model, prepare_train_envs, prepare_test_envs, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.evaluation.evaluator import Evaluator_Feat, Evaluator_Coverage_Count, Evaluator_User_Experience, save_model_fn
from core.evaluation.loggers import LoggerEval_Policy
from core.util.data import get_common_args, get_val_data, get_training_item_domination, get_item_similarity, get_item_popularity
from core.collector.collector import Collector
from core.policy.dqn import DQNPolicy_with_Embedding
from core.trainer.offpolicy import offpolicy_trainer


from tianshou.data import VectorReplayBuffer, PrioritizedVectorReplayBuffer

from tianshou.utils.net.common import Net

# from util.upload import my_upload
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_DQN():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DQN")

    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--reward-normalization', action="store_true", default=False)
    parser.add_argument('--is-double', type=bool, default=True)
    parser.add_argument('--clip-loss-grad', action="store_true", default=False)

    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--training-num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--prioritized-replay', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)

    parser.add_argument("--read_message", type=str, default="UM")
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
    policy = DQNPolicy_with_Embedding(  ## TODO
        net,
        optim,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        reward_normalization=args.reward_normalization,
        is_double=args.is_double, 
        clip_loss_grad=args.clip_loss_grad, 
        
    )
    policy.set_eps(args.eps)  ## args.eps_test

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
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
        force_length=args.step_per_collect / args.training_num,
    )
    # test_collector = Collector(
    #     policy, test_envs_dict,
    #     VectorReplayBuffer(args.buffer_size, len(test_envs)),
    #     preprocess_fn=state_tracker.build_state,
    #     exploration_noise=args.exploration_noise,
    # )
    policy.set_collector(train_collector)  ## TODO
    # train_collector.collect(n_step=args.batch_size * args.training_num)  ## TODO

    test_collector_set = CollectorSet(policy, test_envs_dict, args.buffer_size, args.test_num,
                                      preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return policy, train_collector, test_collector_set, optim


def learn_policy(args, env, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path):
    # log
    # log_path = os.path.join(args.logdir, args.env, 'dqn')
    # writer = SummaryWriter(log_path)
    # logger = TensorboardLogger(writer)
    # def save_best_fn(policy):
        # torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # env = test_collector_set.env
    df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)
    item_feat_domination = get_training_item_domination(args.env)
    item_similarity, item_popularity = get_item_similarity(args.env), get_item_popularity(args.env)

    # set metrics and related evaluator
    metrics = ['len_tra', 'R_tra', 'ctr', 'CV', 'CV_turn', 'ifeat_', 'Diversity', 'Novelty']

    policy.callbacks = [
        Evaluator_Feat(test_collector_set, df_item_val, args.need_transform, item_feat_domination,
                                lbe_item=env.lbe_item if args.need_transform else None, top_rate=args.top_rate, draw_bar=args.draw_bar),
        Evaluator_Coverage_Count(test_collector_set, df_item_val, args.need_transform),
        Evaluator_User_Experience(test_collector_set, df_item_val, item_similarity, item_popularity,
                                  args.need_transform, lbe_item=env.lbe_item if args.need_transform else None),
        LoggerEval_Policy(args.force_length, metrics)]
    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector_set,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,  ## yyq
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,  ## yyq
        # stop_fn=stop_fn,
        # save_best_fn=save_best_fn,
        # logger=logger,
        save_model_fn=functools.partial(save_model_fn,
                                        model_save_path=model_save_path,
                                        state_tracker=state_tracker,
                                        optim=optim,
                                        is_save=args.is_save)
    )

    print(__file__)
    pprint.pprint(result)
    logger.info(result)

def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, train_envs = prepare_train_envs(args, ensemble_models)
    test_envs_dict = prepare_test_envs(args)
    
    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path)


if __name__ == "__main__":
    args_all = get_args_all()
    args = get_common_args(args_all)
    args_DQN = get_args_DQN()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_DQN.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
