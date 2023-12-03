import datetime
import functools
import json
import os
import pickle
import pprint
import random
import socket
import sys
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from core.state_tracker.Caser import StateTracker_Caser
from core.state_tracker.Average import StateTrackerAvg
from core.state_tracker.GRU import StateTracker_GRU
from core.state_tracker.NextItNet import StateTracker_NextItNet
from core.state_tracker.SASRec import StateTracker_SASRec
from core.util.inputs import get_dataset_columns
from core.userModel.user_model_ensemble import EnsembleModel
from environments.Simulated_Env.base import BaseSimulatedEnv

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.util.data import get_true_env
from core.evaluation.evaluator import Evaluator_Feat, Evaluator_Coverage_Count, Evaluator_User_Experience, save_model_fn
from core.evaluation.loggers import LoggerEval_Policy

from tianshou.data.buffer.base import ReplayBuffer
from tianshou.data import VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.trainer.offline import offline_trainer

from core.util.utils import create_dir
import logzero
from logzero import logger

def get_args_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    
    # training
    parser.add_argument('--remove_recommended_ids', action="store_true", default=False)

    parser.add_argument('--is_draw_bar', dest='draw_bar', action='store_true')
    parser.add_argument('--no_draw_bar', dest='draw_bar', action='store_false')
    parser.set_defaults(draw_bar=False)

    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')
    parser.set_defaults(is_userinfo=False)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.set_defaults(cpu=False)

    parser.add_argument('--is_save', dest='is_save', action='store_true')
    parser.add_argument('--no_save', dest='is_save', action='store_false')
    parser.set_defaults(is_save=False)

    parser.add_argument('--is_use_userEmbedding', dest='use_userEmbedding', action='store_true')
    parser.add_argument('--no_use_userEmbedding', dest='use_userEmbedding', action='store_false')
    parser.set_defaults(use_userEmbedding=False)

    parser.add_argument('--is_use_pretrained_embedding', dest='use_pretrained_embedding', action='store_true')
    parser.add_argument('--no_use_pretrained_embedding', dest='use_pretrained_embedding', action='store_false')
    parser.set_defaults(use_pretrained_embedding=True)

    parser.add_argument('--is_exploration_noise', dest='exploration_noise', action='store_true')
    parser.add_argument('--no_exploration_noise', dest='exploration_noise', action='store_false')
    parser.set_defaults(exploration_noise=False)
    parser.add_argument('--eps', default=0.2, type=float)

    parser.add_argument('--is_freeze_emb', dest='freeze_emb', action='store_true')
    parser.add_argument('--no_freeze_emb', dest='freeze_emb', action='store_false')
    parser.set_defaults(freeze_emb=False)

    # state tracker
    parser.add_argument('--reward_handle', type=str, default='cat')  # in {"no", "cat", "cat2", "mul"}
    parser.add_argument("--which_tracker", type=str, default="avg")  # in {"avg", "caser", "sasrec", "gru", "nextitnet"}

    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument('--window_size', default=3, type=int)

    parser.add_argument('--is_random_init', dest='random_init', action='store_true')
    parser.add_argument('--no_random_init', dest='random_init', action='store_false')
    parser.set_defaults(random_init=False)

    # State_tracker Caser
    parser.add_argument('--filter_sizes', type=int, nargs='*', default=[2, 3, 4])
    parser.add_argument("--num_filters", type=int, default=16)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    # State_tracker SASRec
    # parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=1)
    # State_tracker nextitnet
    parser.add_argument("--dilations", type=str, default='[1, 2, 1, 2, 1, 2]')

    # tianshou
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])

    parser.add_argument('--episode-per-collect', type=int, default=100)
    parser.add_argument('--training-num', type=int, default=100)
    parser.add_argument('--test-num', type=int, default=100)

    parser.add_argument('--render', type=float, default=0)
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')

    parser.add_argument("--read_message", type=str, default="UM")

    args = parser.parse_known_args()[0]
    return args


def prepare_dir_log(args):
    # %% 1. Create dirs
    MODEL_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.model_name)
    create_dirs = [os.path.join(".", "saved_models"),
                   os.path.join(".", "saved_models", args.env),
                   MODEL_SAVE_PATH,
                   os.path.join(MODEL_SAVE_PATH, "logs")]
    create_dir(create_dirs)

    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)
    hostname = socket.gethostname()
    args.hostname = hostname
    logger.info(json.dumps(vars(args), indent=2))

    return MODEL_SAVE_PATH, logger_path


def prepare_user_model(args):
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)

    UM_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.user_model_name)
    # MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{args.read_message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(UM_SAVE_PATH, "params", f"[{args.read_message}]_params.pickle")

    with open(MODEL_PARAMS_PATH, "rb") as file:
        model_params = pickle.load(file)

    n_models = model_params["n_models"]
    model_params.pop('n_models')

    ensemble_models = EnsembleModel(n_models, args.read_message, UM_SAVE_PATH, **model_params)
    ensemble_models.load_all_models()

    # user_model = ensemble_models.user_models[0]
    # if hasattr(user_model, 'ab_embedding_dict') and args.is_ab:
    #     alpha_u = user_model.ab_embedding_dict["alpha_u"].weight.detach().cpu().numpy()
    #     beta_i = user_model.ab_embedding_dict["beta_i"].weight.detach().cpu().numpy()
    # else:
    #     print("Note there are no available alpha and beta！！")
    # alpha_u = None
    # beta_i = None
    # return ensemble_models, alpha_u, beta_i
    return ensemble_models

def evenly_distribute_trajectories_to_bins(df_user_num_mapped, num_bins):
    df_user_num_sorted = df_user_num_mapped.sort_values("item_id", ascending=False)

    bins = np.zeros([num_bins])
    bins_ind = defaultdict(set)
    for user, num in df_user_num_sorted.reset_index().to_numpy():
        ind = bins.argmin()
        bins_ind[ind].add(user)
        bins[ind] += num
        # np.zeros([num_bins])

    max_size = max(bins)
    buffer_size = max_size * num_bins
    return bins_ind, max_size, buffer_size

def construct_buffer_from_offline_data(args, df_train, env):
    num_bins = args.test_num

    df_user_num = df_train[["user_id", "item_id"]].groupby("user_id").agg(len)
    

    if args.env == 'KuaiEnv-v0':
        assert hasattr(env, "lbe_user")
        df_user_num_mapped = df_user_num.loc[env.lbe_user.classes_]
        df_user_num_mapped = df_user_num_mapped.reset_index(drop=True)
        assert len(env.mat) == len(df_user_num_mapped)

        assert hasattr(env, "lbe_item")
        df_numpy = df_train[["user_id", "item_id", args.yfeat]].to_numpy()
        
        func = np.vectorize(lambda x: x in env.lbe_item.classes_)
        indices_valid_item = func(df_numpy[:,1].astype(int))
        
        df_train_part = df_train[["user_id", "item_id", args.yfeat]].loc[indices_valid_item]
        # df_train_part["user_id"] = dummy_user = 0  # set to dummy user. Since these users are not in the evaluational environment.
        df_train_part = df_train_part.reset_index(drop=True)
        # df_user_items = df_train_part.groupby("user_id").agg(list)

        df_train_part["item_id"] = env.lbe_item.transform(df_train_part["item_id"])

        func_replace_user = np.vectorize(lambda x: x if x in env.lbe_user.classes_ else np.random.choice(env.lbe_user.classes_))
        users_replaced = func_replace_user(df_train_part["user_id"].to_numpy().astype(int))

        df_train_part["user_id"] = env.lbe_user.transform(users_replaced)

    elif args.env == 'YahooEnv-v0':
        df_user_num_mapped = df_user_num.iloc[:len(env.mat)]
        df_train_part = df_train[["user_id", "item_id", args.yfeat]]

    else:  # KuaiRand-v0 and CoatEnv-v0
        df_user_num_mapped = df_user_num
        df_train_part = df_train[["user_id", "item_id", args.yfeat]]


    MIN = df_train[args.yfeat].min()
    MAX = df_train[args.yfeat].max()
    Max_Min_Scaler = lambda x : (x-MIN)/(MAX-MIN)
    df_train_part[args.yfeat] = df_train_part[args.yfeat].apply(Max_Min_Scaler)
    
    bins_ind, max_size, buffer_size = evenly_distribute_trajectories_to_bins(df_user_num_mapped, num_bins)
    buffer = VectorReplayBuffer(buffer_size, num_bins)

    # env.max_turn = max_size

    # df_user_items = df_train_part[["user_id", "item_id", args.yfeat]].groupby("user_id").agg(list)

    df_user_items = df_train_part.groupby("user_id").agg(list)
    for indices, users in tqdm(bins_ind.items(), total=len(bins_ind), desc="preparing offline data into buffer..."):
        for user in users:
            (user_reset, item_reset), info = env.reset()
            env.cur_user = user
            # env.state = np.array([user, item_reset])
            # env.action

            # items = [item_reset] + df_user_items.loc[user][0]
            items = df_user_items.loc[user]["item_id"]
            rewards = df_user_items.loc[user][args.yfeat]
            np_ui_pair = np.vstack([np.ones_like(items) * user, items]).T

            terminateds = np.zeros(len(rewards), dtype=bool)
            truncateds = np.zeros(len(rewards), dtype=bool)
            rew_prevs = [0] + rewards[:-1]
            is_starts = np.zeros(len(rewards), dtype=bool)
            obs_items = np.zeros(len(rewards), dtype=int)
            # obs_next_items = np.zeros(len(rewards), dtype=int)
            
            set_is_start = True

            for k, item in enumerate(items):
                if set_is_start:
                    is_starts[k] = True
                    obs_items[k] = item_reset
                    rew_prevs[k] = 0 
                    set_is_start = False
                else:
                    obs_items[k] = items[k-1]
                    assert rew_prevs[k] == rewards[k-1]
                    
                obs_next, rew, terminated, truncated, info = env.step(item)
                if terminated or truncated:  
                    (user_reset, item_reset), info = env.reset()
                    set_is_start = True
                    # env.cur_user = user_reset
                terminateds[k] = terminated
                truncateds[k] = truncated
                # print(env.cur_user, obs_next, rew, done, info)

            terminateds[-1] = True
            
            obs = np.vstack([np.ones_like(items) * user, obs_items]).T

            batch = Batch(obs=obs, obs_next=np_ui_pair, act=items, is_start=is_starts,
                          policy={}, info={}, rew=rewards, rew_prev=rew_prevs, terminated=terminateds, truncated=truncateds)
            ptr, ep_rew, ep_len, ep_idx = buffer.add(batch, buffer_ids=np.ones([len(batch)], dtype=int) * indices)

    return buffer


def prepare_buffer_via_offline_data(args):
    env, dataset, kwargs_um = get_true_env(args)
    df_train, df_user, df_item, list_feat = dataset.get_train_data()
    if "time_ms" in df_train.columns:
        df_train.rename(columns={"time_ms": "timestamp"}, inplace=True)
        df_train = df_train.sort_values(["user_id", "timestamp"])
    if not "timestamp" in df_train.columns:
        df_train = df_train.sort_values(["user_id"])

    df_train[["user_id", "item_id"]].to_numpy()

    buffer = construct_buffer_from_offline_data(args, df_train, env)
    env.max_turn = args.max_turn

    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    # seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    return env, dataset, buffer

def prepare_train_envs(args, ensemble_models):
    env, dataset, kwargs_um = get_true_env(args)

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,
    }

    train_envs = DummyVectorEnv(
        [lambda: BaseSimulatedEnv(**kwargs) for _ in range(args.training_num)])
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)

    return env, dataset, train_envs


def prepare_test_envs(args):
    env, dataset, kwargs_um = get_true_env(args)
    env_task_class = type(env)
    test_envs = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_0 = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_x = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])

    test_envs_dict = {"FB": test_envs, "NX_0": test_envs_NX_0, f"NX_{args.force_length}": test_envs_NX_x}

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    return test_envs_dict



def setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict, use_buffer_in_train=False):
    if use_buffer_in_train:
        buffer = train_envs
        train_envs = None

    saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)
    if args.use_pretrained_embedding:
    # if args.which_tracker.lower() == "avg":
        user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
            get_dataset_columns(saved_embedding["feat_user"].weight.shape[1], saved_embedding["feat_item"].weight.shape[1],
                                env.mat.shape[0], env.mat.shape[1], envname=args.env)
    else:
        user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
            get_dataset_columns(args.embedding_dim, args.embedding_dim, env.mat.shape[0], env.mat.shape[1], envname=args.env)

    args.action_shape = action_columns[0].vocabulary_size
    args.state_dim = action_columns[0].embedding_dim

    if args.use_userEmbedding:
        args.state_dim = action_columns[0].embedding_dim + saved_embedding.feat_user.weight.shape[1]

    if use_buffer_in_train:
        train_max = buffer.rew.max()
        train_min = buffer.rew.min()
    else:
        train_max = train_envs.get_env_attr("MAX_R")[0] - train_envs.get_env_attr("MIN_R")[0]
        train_min = 0
    test_max = test_envs_dict['FB'].get_env_attr("mat")[0].max()
    test_min = test_envs_dict['FB'].get_env_attr("mat")[0].min()

    if args.which_tracker.lower() == "caser":
        assert args.window_size >= max(args.filter_sizes)
        state_tracker = StateTracker_Caser(user_columns, action_columns, feedback_columns, args.state_dim,
                                           train_max, train_min, test_max, test_min, reward_handle=args.reward_handle, saved_embedding=saved_embedding,
                                           device=args.device,
                                           window_size=args.window_size,
                                           filter_sizes=args.filter_sizes, num_filters=args.num_filters,
                                           dropout_rate=args.dropout_rate).to(args.device)
    elif args.which_tracker.lower() == "gru":
        state_tracker = StateTracker_GRU(user_columns, action_columns, feedback_columns, args.state_dim,
                                         train_max, train_min, test_max, test_min, reward_handle=args.reward_handle, saved_embedding=saved_embedding,
                                         device=args.device,
                                         window_size=args.window_size).to(args.device)
    elif args.which_tracker.lower() == "sasrec":
        state_tracker = StateTracker_SASRec(user_columns, action_columns, feedback_columns, args.state_dim,
                                            train_max, train_min, test_max, test_min, reward_handle=args.reward_handle, saved_embedding=saved_embedding,
                                            device=args.device, window_size=args.window_size,
                                            dropout_rate=args.dropout_rate, num_heads=args.num_heads).to(args.device)
    elif args.which_tracker.lower() == "nextitnet":
        state_tracker = StateTracker_NextItNet(user_columns, action_columns, feedback_columns, args.state_dim,
                                               train_max, train_min, test_max, test_min, reward_handle=args.reward_handle, saved_embedding=saved_embedding,
                                               device=args.device, window_size=args.window_size,
                                               dilations=args.dilations).to(args.device)
    elif args.which_tracker.lower() == "avg":
        assert args.use_pretrained_embedding
        state_tracker = StateTrackerAvg(user_columns, action_columns, feedback_columns, args.state_dim,
                                         train_max, train_min, test_max, test_min, reward_handle=args.reward_handle, saved_embedding=saved_embedding,
                                         device=args.device, window_size=args.window_size,
                                         use_userEmbedding=args.use_userEmbedding).to(args.device)
    else:
        return None
    
    args.state_dim = state_tracker.final_dim
    

    return state_tracker


def learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="onpolicy"):  # , is_onpolicy=True, is_offline=False
    # log
    # log_path = os.path.join(args.logdir, args.env, 'a2c')
    # writer = SummaryWriter(log_path)
    # logger1 = TensorboardLogger(writer)
    # def save_best_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # Evaluation
    df_val, df_user_val, df_item_val, list_feat = dataset.get_val_data()
    item_feat_domination = dataset.get_domination()
    item_similarity, item_popularity = dataset.get_item_similarity(), dataset.get_item_popularity()

    metrics = ['len_tra', 'R_tra', 'ctr', 'CV', 'CV_turn', 'ifeat_', 'Diversity', 'Novelty']

    policy.callbacks = [
        Evaluator_Feat(test_collector_set, df_item_val, args.need_transform, item_feat_domination,
                                lbe_item=env.lbe_item if args.need_transform else None, top_rate=args.top_rate, draw_bar=args.draw_bar),
        Evaluator_Coverage_Count(test_collector_set, df_item_val, args.need_transform),
        Evaluator_User_Experience(test_collector_set, df_item_val, item_similarity, item_popularity,
                                  args.need_transform, lbe_item=env.lbe_item if args.need_transform else None),
        LoggerEval_Policy(args.force_length, metrics)]
    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    if trainer == "offline":
        buffer = train_collector
        train_collector = None
        assert isinstance(buffer, ReplayBuffer)

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

    elif trainer == "onpolicy":
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector_set,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            episode_per_collect=args.episode_per_collect,
            # stop_fn=stop_fn,
            # save_best_fn=save_best_fn,
            # logger=logger1,
            save_model_fn=functools.partial(save_model_fn,
                                            model_save_path=model_save_path,
                                            state_tracker=state_tracker,
                                            optim=optim,
                                            is_save=args.is_save)
        )

    elif trainer == "offpolicy":
        result = offpolicy_trainer(	
            policy,	
            train_collector,	
            test_collector_set,	
            args.epoch,	
            args.step_per_epoch,	
            args.step_per_collect,
            args.test_num,	
            args.batch_size,	
            update_per_step=args.update_per_step,
            # stop_fn=stop_fn,	
            # save_best_fn=save_best_fn,	
            # logger=logger1,	
            save_model_fn=functools.partial(save_model_fn,	
                                            model_save_path=model_save_path,	
                                            state_tracker=state_tracker,	
                                            optim=optim,	
                                            is_save=args.is_save)
        )

    else:
        raise ValueError("Unexpected Trainer")

    print(__file__)
    pprint.pprint(result)
    logger.info(result)