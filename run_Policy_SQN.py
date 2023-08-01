import argparse
import datetime
import functools
import json
import os
import pprint

import traceback

import torch


import sys


from policy_utils import prepare_dir_log, prepare_buffer_via_offline_data, setup_offline_state_tracker
from run_Policy_Main import get_args_all

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.collector_set import CollectorSet
from core.evaluation.evaluator import Callback_Coverage_Count
from core.util.layers import Actor_Linear
from core.policy.sqn import SQN
from core.trainer.offline import offline_trainer
from core.configs import get_common_args, get_val_data, get_training_item_domination
from core.util.inputs import get_dataset_columns


from tianshou.utils.net.common import ActorCritic

# from util.upload import my_upload
from util.utils import LoggerCallback_Policy, save_model_fn
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_SQN():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SQN")
    parser.add_argument('--which_head', type=str, default='qhead')  # in {"shead", "qhead", "bcq"}

    # bcq
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.6)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)
    parser.add_argument("--eps-test", type=float, default=0.001)
    # parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)

    # parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="SQN")

    args = parser.parse_known_args()[0]
    return args

    # %% 4. Setup model


def setup_policy_model(args, state_tracker, buffer, test_envs_dict):
    # # ensemble_models, _, _ = prepare_user_model(args)
    #
    # # saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)
    # user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
    #     get_dataset_columns(args.embedding_dim, args.embedding_dim, env.mat.shape[0], env.mat.shape[1],
    #                         envname=args.env)
    #
    # args.action_shape = action_columns[0].vocabulary_size
    # args.state_dim = action_columns[0].embedding_dim
    #
    # args.max_action = env.action_space.high[0]
    #
    # if args.which_tracker.lower() == "caser":
    #     state_tracker = StateTracker_Caser(user_columns, action_columns, feedback_columns, args.state_dim,
    #                                        device=args.device,
    #                                        window_size=args.window_size,
    #                                        filter_sizes=args.filter_sizes, num_filters=args.num_filters,
    #                                        dropout_rate=args.dropout_rate).to(args.device)
    # elif args.which_tracker.lower() == "gru":
    #     state_tracker = StateTracker_GRU(user_columns, action_columns, feedback_columns, args.state_dim,
    #                                      device=args.device,
    #                                      window_size=args.window_size).to(args.device)
    # elif args.which_tracker.lower() == "sasrec":
    #     state_tracker = StateTracker_SASRec(user_columns, action_columns, feedback_columns, args.state_dim,
    #                                         device=args.device, window_size=args.window_size,
    #                                         dropout_rate=args.dropout_rate, num_heads=args.num_heads).to(args.device)
    #
    # # state_tracker = state_tracker.to(args.device)

    model_final_layer = Actor_Linear(args.state_dim, args.action_shape, device=args.device).to(args.device)
    imitation_final_layer = Actor_Linear(args.state_dim, args.action_shape, device=args.device).to(args.device)

    actor_critic = ActorCritic(model_final_layer, imitation_final_layer)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    policy = SQN(
        model_final_layer,
        imitation_final_layer,
        optim,
        args.gamma,
        args.n_step,
        args.target_update_freq,
        args.eps_test,
        args.unlikely_action_threshold,
        args.imitation_logits_penalty,
        state_tracker=state_tracker,
        buffer=buffer,
        which_head=args.which_head
    )


    test_collector_set = CollectorSet(policy, test_envs_dict, args.buffer_size, args.test_num,
                                      preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return policy, test_collector_set, optim


def learn_policy(args, env, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path):
    # log
    # t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # log_file = f'seed_{args.seed}_{t0}-{args.env.replace("-", "_")}_bcq'
    # log_path = os.path.join(args.logdir, args.env, 'bcq', log_file)
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(args))
    # logger1 = TensorboardLogger(writer)
    #
    # def save_best_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)
    item_feat_domination = get_training_item_domination(args.env)
    policy.callbacks = [
        Callback_Coverage_Count(test_collector_set, df_item_val, args.need_transform, item_feat_domination,
                                lbe_item=env.lbe_item if args.need_transform else None, top_rate=args.top_rate, draw_bar=args.draw_bar),
        LoggerCallback_Policy(logger_path, args.force_length)]
    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))
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
        # # stop_fn=stop_fn,
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
    env, buffer, test_envs_dict = prepare_buffer_via_offline_data(args)

    # %% 3. Setup policy
    state_tracker = setup_offline_state_tracker(args, env, buffer, test_envs_dict)
    policy, test_collector_set, optim = setup_policy_model(args, state_tracker, buffer, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_common_args(args_all)
    args_SQN = get_args_SQN()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_SQN.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
