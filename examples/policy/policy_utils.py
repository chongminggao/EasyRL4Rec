import datetime
import json
import os
import pickle
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
from environments.simulated_env import SimulatedEnv

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.util.data import get_training_data, get_true_env

from tianshou.data import VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv

from core.util.utils import create_dir
import logzero
from logzero import logger

def get_args_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--cuda', default=0, type=int)

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

    parser.add_argument('--is_exploration_noise', dest='exploration_noise', action='store_true')
    parser.add_argument('--no_exploration_noise', dest='exploration_noise', action='store_false')
    parser.set_defaults(exploration_noise=False)
    parser.add_argument('--eps', default=0.1, type=float)

    parser.add_argument('--is_freeze_emb', dest='freeze_emb', action='store_true')
    parser.add_argument('--no_freeze_emb', dest='freeze_emb', action='store_false')
    parser.set_defaults(freeze_emb=False)

    # state tracker
    parser.add_argument('--reward_handle', type=str, default='cat')  # in {"no", "cat", "cat2", "mul"}
    parser.add_argument("--which_tracker", type=str, default="avg")  # in {"avg", "caser", "sasrec", "gru", "nextitnet"}

    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument('--window_size', default=3, type=int)

    # State_tracker Caser
    parser.add_argument('--filter_sizes', type=int, nargs='*', default=[2, 3, 4])
    parser.add_argument("--num_filters", type=int, default=16)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    # State_tracker SASRec
    # parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=1)
    # State_tracker nextitnet
    parser.add_argument("--dilations", type=str, default='[1, 2, 1, 2, 1, 2]')

    # Env
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument('--tau', default=0, type=float)
    parser.add_argument('--gamma_exposure', default=10, type=float)

    parser.add_argument('--lambda_variance', default=0.05, type=float)
    parser.add_argument('--lambda_entropy', default=5, type=float)

    parser.add_argument('--is_exposure_intervention', dest='use_exposure_intervention', action='store_true')
    parser.add_argument('--no_exposure_intervention', dest='use_exposure_intervention', action='store_false')
    parser.set_defaults(use_exposure_intervention=False)

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
        indices = [False] * len(df_numpy)
        for k, (user, item, yfeat) in tqdm(enumerate(df_numpy), total=len(df_numpy)):
            if int(item) in env.lbe_item.classes_:
                indices[k] = True
        df_filtered = df_train[["user_id", "item_id", args.yfeat]].loc[indices]
        df_filtered[
            "user_id"] = dummy_user = 0  # set to dummy user. Since these users are not in the evaluational environment.
        df_filtered = df_filtered.reset_index(drop=True)
        # df_user_items = df_filtered.groupby("user_id").agg(list)

        df_filtered["item_id"] = env.lbe_item.transform(df_filtered["item_id"])

        num_each = int(np.ceil(len(df_filtered) / num_bins))
        env.max_turn = num_each
        buffer_size = num_each * num_bins
        buffer = VectorReplayBuffer(buffer_size, num_bins)

        ind_pair = zip(np.arange(0, buffer_size, num_each), np.arange(num_each, buffer_size + num_each, num_each))
        for ind_buffer, (left, right) in tqdm(enumerate(ind_pair), total=num_bins,
                                              desc="preparing offline data into buffer..."):
            seq = df_filtered.iloc[int(left):int(right)]

            items = [-1] + seq["item_id"].to_list()
            rewards = seq[args.yfeat].to_numpy()
            np_ui_pair = np.vstack([np.ones_like(items) * dummy_user, items]).T

            env.reset()
            env.cur_user = dummy_user
            dones = np.zeros(len(rewards), dtype=bool)

            for k, item in enumerate(items[1:]):
                obs_next, rew, done, info = env.step(item)
                if done:
                    env.reset()
                    env.cur_user = dummy_user
                dones[k] = done
                dones[-1] = True
                # print(env.cur_user, obs_next, rew, done, info)

            batch = Batch(obs=np_ui_pair[:-1], obs_next=np_ui_pair[1:], act=items[1:],
                          policy={}, info={}, rew=rewards, done=dones)

            ptr, ep_rew, ep_len, ep_idx = buffer.add(batch, buffer_ids=np.ones([len(batch)], dtype=int) * ind_buffer)

        return buffer

    elif args.env == 'YahooEnv-v0':
        df_user_num_mapped = df_user_num.iloc[:len(env.mat)]
    else:  # KuaiRand-v0 and CoatEnv-v0
        df_user_num_mapped = df_user_num

    df_user_num_sorted = df_user_num_mapped.sort_values("item_id", ascending=False)

    bins = np.zeros([num_bins])
    bins_ind = defaultdict(set)
    for user, num in df_user_num_sorted.reset_index().to_numpy():
        ind = bins.argmin()
        bins_ind[ind].add(user)
        bins[ind] += num
        np.zeros([num_bins])

    max_size = max(bins)
    buffer_size = max_size * num_bins
    buffer = VectorReplayBuffer(buffer_size, num_bins)

    # env, env_task_class, kwargs_um = get_true_env(args)
    env.max_turn = max_size
    df_user_items = df_train[["user_id", "item_id", args.yfeat]].groupby("user_id").agg(list)
    for indices, users in tqdm(bins_ind.items(), total=len(bins_ind), desc="preparing offline data into buffer..."):
        for user in users:
            items = [-1] + df_user_items.loc[user][0]
            rewards = df_user_items.loc[user][1]
            np_ui_pair = np.vstack([np.ones_like(items) * user, items]).T

            env.reset()
            env.cur_user = user
            dones = np.zeros(len(rewards), dtype=bool)

            for k, item in enumerate(items[1:]):
                obs_next, rew, done, info = env.step(item)
                if done:
                    env.reset()
                    env.cur_user = user
                dones[k] = done
                dones[-1] = True
                # print(env.cur_user, obs_next, rew, done, info)
            batch = Batch(obs=np_ui_pair[:-1], obs_next=np_ui_pair[1:], act=items[1:],
                          policy={}, info={}, rew=rewards, done=dones)
            ptr, ep_rew, ep_len, ep_idx = buffer.add(batch, buffer_ids=np.ones([len(batch)], dtype=int) * indices)

    return buffer


def prepare_buffer_via_offline_data(args):
    df_train, df_user, df_item, list_feat = get_training_data(args.env)
    # df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)
    # df_train = df_train.head(10000)
    if "time_ms" in df_train.columns:
        df_train.rename(columns={"time_ms": "timestamp"}, inplace=True)
        df_train = df_train.sort_values(["user_id", "timestamp"])
    if not "timestamp" in df_train.columns:
        df_train = df_train.sort_values(["user_id"])

    df_train[["user_id", "item_id"]].to_numpy()

    env, env_task_class, kwargs_um = get_true_env(args)
    buffer = construct_buffer_from_offline_data(args, df_train, env)
    env.max_turn = args.max_turn

    test_envs = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_0 = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_x = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])

    test_envs_dict = {"FB": test_envs, "NX_0": test_envs_NX_0, f"NX_{args.force_length}": test_envs_NX_x}

    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    # seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    return env, buffer, test_envs_dict

def prepare_envs(args, ensemble_models, alpha_u=None, beta_i=None):
    env, env_task_class, kwargs_um = get_true_env(args)

    # user_features, item_features, reward_features = get_features(args.env, args.is_userinfo)
    # embedding_dim = ensemble_models.user_models[0].feature_columns[0].embedding_dim

    # dataset_val, df_user_val, df_item_val = load_dataset_val(args, user_features, item_features, reward_features, embedding_dim, embedding_dim)

    # entropy_user, map_entropy = ensemble_models.get_save_entropy_mat(args.env, args.entropy_window)

    entropy_dict = dict()
    # if 0 in args.entropy_window:
    #     entropy_path = os.path.join(ensemble_models.Entropy_PATH, "user_entropy.csv")
    #     entropy = pd.read_csv(entropy_path)
    #     entropy.set_index("user_id", inplace=True)
    #     entropy_mat_0 = entropy.to_numpy().reshape([-1])
    #     entropy_dict.update({"on_user": entropy_mat_0})
    if len(set(args.entropy_window) - set([0])):
        savepath = os.path.join(ensemble_models.Entropy_PATH, "map_entropy.pickle")
        map_entropy = pickle.load(open(savepath, 'rb'))
        entropy_dict.update({"map": map_entropy})

    entropy_set = set(args.entropy_window)
    entropy_min = 0
    entropy_max = 0
    if len(entropy_set):
        for entropy_term in entropy_set:
            entropy_min += min([v for k, v in entropy_dict["map"].items() if len(k) == entropy_term])
            entropy_max += max([v for k, v in entropy_dict["map"].items() if len(k) == entropy_term])

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    with open(ensemble_models.VAR_MAT_PATH, "rb") as file:
        maxvar_mat = pickle.load(file)

    kwargs = {
        "ensemble_models": ensemble_models,
        # "dataset_val": dataset_val,
        # "need_transform": args.need_transform,
        "env_task_class": env_task_class,
        # "user_model": user_model,
        "use_exposure_intervention": args.use_exposure_intervention,
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "version": args.version,
        "tau": args.tau,
        "alpha_u": alpha_u,
        "beta_i": beta_i,
        "lambda_entropy": args.lambda_entropy,
        "lambda_variance": args.lambda_variance,
        "predicted_mat": predicted_mat,
        "maxvar_mat": maxvar_mat,
        "entropy_dict": entropy_dict,
        "entropy_window": args.entropy_window,
        "gamma_exposure": args.gamma_exposure,
        "step_n_actions": max(args.entropy_window) if len(args.entropy_window) else 0,
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
    }

    # simulatedEnv = SimulatedEnv(**kwargs)

    train_envs = DummyVectorEnv(
        [lambda: SimulatedEnv(**kwargs) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_0 = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_x = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])

    test_envs_dict = {"FB": test_envs, "NX_0": test_envs_NX_0, f"NX_{args.force_length}": test_envs_NX_x}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    return env, train_envs, test_envs_dict


def setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict):
    saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)
    if args.which_tracker.lower() == "avg":
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

    train_max = train_envs.get_env_attr("MAX_R")[0] - train_envs.get_env_attr("MIN_R")[0]
    train_min = 0
    test_max = test_envs_dict['FB'].get_env_attr("mat")[0].max()
    test_min = test_envs_dict['FB'].get_env_attr("mat")[0].min()

    if args.which_tracker.lower() == "caser":
        state_tracker = StateTracker_Caser(user_columns, action_columns, feedback_columns, args.state_dim,
                                           device=args.device,
                                           window_size=args.window_size,
                                           filter_sizes=args.filter_sizes, num_filters=args.num_filters,
                                           dropout_rate=args.dropout_rate).to(args.device)
        args.state_dim = state_tracker.final_dim
    elif args.which_tracker.lower() == "gru":
        state_tracker = StateTracker_GRU(user_columns, action_columns, feedback_columns, args.state_dim,
                                         device=args.device,
                                         window_size=args.window_size).to(args.device)
        args.state_dim = state_tracker.final_dim
    elif args.which_tracker.lower() == "sasrec":
        state_tracker = StateTracker_SASRec(user_columns, action_columns, feedback_columns, args.state_dim,
                                            device=args.device, window_size=args.window_size,
                                            dropout_rate=args.dropout_rate, num_heads=args.num_heads).to(args.device)
        args.state_dim = state_tracker.final_dim
    elif args.which_tracker.lower() == "nextitnet":
        state_tracker = StateTracker_NextItNet(user_columns, action_columns, feedback_columns, args.state_dim,
                                               device=args.device, window_size=args.window_size,
                                               dilations=args.dilations).to(args.device)
        args.state_dim = state_tracker.final_dim
    elif args.which_tracker.lower() == "avg":
        state_tracker = StateTrackerAvg(user_columns, action_columns, feedback_columns, args.state_dim,
                                         saved_embedding,
                                         train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                         device=args.device, window_size=args.window_size,
                                         use_userEmbedding=args.use_userEmbedding).to(args.device)
        if args.reward_handle == "cat" or args.reward_handle == "cat2":
            args.state_dim += 1
    else:
        return None

    return state_tracker


def setup_offline_state_tracker(args, ensemble_models, env, buffer, test_envs_dict):
    saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)
    if args.which_tracker.lower() == "avg":
        user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
            get_dataset_columns(saved_embedding["feat_user"].weight.shape[1],
                                saved_embedding["feat_item"].weight.shape[1],
                                env.mat.shape[0], env.mat.shape[1], envname=args.env)
    else:
        user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
            get_dataset_columns(args.embedding_dim, args.embedding_dim, env.mat.shape[0], env.mat.shape[1],
                                envname=args.env)

    args.action_shape = action_columns[0].vocabulary_size
    args.state_dim = action_columns[0].embedding_dim

    if args.use_userEmbedding:
        args.state_dim = action_columns[0].embedding_dim + saved_embedding.feat_user.weight.shape[1]

    train_max = buffer.rew.max()
    train_min = buffer.rew.min()
    test_max = test_envs_dict['FB'].get_env_attr("mat")[0].max()
    test_min = test_envs_dict['FB'].get_env_attr("mat")[0].min()

    if args.which_tracker.lower() == "caser":
        state_tracker = StateTracker_Caser(user_columns, action_columns, feedback_columns, args.state_dim,
                                           device=args.device,
                                           window_size=args.window_size,
                                           filter_sizes=args.filter_sizes, num_filters=args.num_filters,
                                           dropout_rate=args.dropout_rate).to(args.device)
        args.state_dim = state_tracker.final_dim
    elif args.which_tracker.lower() == "gru":
        state_tracker = StateTracker_GRU(user_columns, action_columns, feedback_columns, args.state_dim,
                                         device=args.device,
                                         window_size=args.window_size).to(args.device)
        args.state_dim = state_tracker.final_dim
    elif args.which_tracker.lower() == "sasrec":
        state_tracker = StateTracker_SASRec(user_columns, action_columns, feedback_columns, args.state_dim,
                                            device=args.device, window_size=args.window_size,
                                            dropout_rate=args.dropout_rate, num_heads=args.num_heads).to(args.device)
        args.state_dim = state_tracker.final_dim
    elif args.which_tracker.lower() == "nextitnet":
        state_tracker = StateTracker_NextItNet(user_columns, action_columns, feedback_columns, args.state_dim,
                                               device=args.device, window_size=args.window_size,
                                               dilations=args.dilations).to(args.device)
        args.state_dim = state_tracker.final_dim
    elif args.which_tracker.lower() == "avg":
        state_tracker = StateTrackerAvg(user_columns, action_columns, feedback_columns, args.state_dim,
                                         saved_embedding,
                                         train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                         device=args.device, window_size=args.window_size,
                                         use_userEmbedding=args.use_userEmbedding).to(args.device)
        if args.reward_handle == "cat" or args.reward_handle == "cat2":
            args.state_dim += 1
    else:
        return None

    return state_tracker


