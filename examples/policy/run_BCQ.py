import argparse
import functools
import os
import pprint
import sys
import traceback

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, prepare_dir_log, prepare_user_model, prepare_buffer_via_offline_data, prepare_test_envs, setup_offline_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.evaluation.evaluator import Evaluator_Feat, Evaluator_Coverage_Count, Evaluator_User_Experience, save_model_fn
from core.evaluation.loggers import LoggerEval_Policy
from core.policy.discrete_bcq import DiscreteBCQPolicy_withEmbedding
from core.trainer.offline import offline_trainer
from core.util.data import get_val_data, get_env_args, \
    get_training_item_domination, get_item_similarity, get_item_popularity

from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor

# from util.upload import my_upload
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_BCQ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BCQ")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.6)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    # parser.add_argument("--update-per-epoch", type=int, default=5000)
    parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="BCQ")

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
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    policy = DiscreteBCQPolicy_withEmbedding(
        policy_net,
        imitation_net,
        optim,
        args.gamma,
        args.n_step,
        args.target_update_freq,
        args.eps_test,
        args.unlikely_action_threshold,
        args.imitation_logits_penalty,
        state_tracker=state_tracker,
        buffer=buffer,
    )

    # collector
    # buffer has been gathered
    # train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    # test_collector = Collector(
    #     policy, test_envs,
    #     VectorReplayBuffer(args.buffer_size, len(test_envs)),
    #     preprocess_fn=state_tracker.build_state,
    #     exploration_noise=args.exploration_noise,
    # )
    test_collector_set = CollectorSet(policy, test_envs_dict, args.buffer_size, args.test_num,
                                    #   preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return policy, test_collector_set, optim


def learn_policy(args, env, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path):
    # log
    # t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # log_file = f'seed_{args.seed}_{t0}-{args.env.replace("-", "_")}_sqn'
    # log_path = os.path.join(args.logdir, args.env, 'sqn', log_file)
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(args))
    # logger1 = TensorboardLogger(writer)
    # def save_best_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

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

    result = offline_trainer(
        policy,
        buffer,
        test_collector_set,
        args.epoch,
        args.step_per_epoch,
        args.test_num,
        args.batch_size,
        # save_best_fn=save_best_fn,
        # stop_fn=stop_fn,
        # logger=logger1,
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
    env, buffer = prepare_buffer_via_offline_data(args)
    test_envs_dict = prepare_test_envs(args)

    # %% 3. Setup policy
    state_tracker = setup_offline_state_tracker(args, ensemble_models, env, buffer, test_envs_dict)
    policy, test_collector_set, optim = setup_policy_model(args, state_tracker, buffer, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path)


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
