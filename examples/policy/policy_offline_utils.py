import argparse
import random
from collections import defaultdict
import math
import numpy as np
import torch
from tqdm import tqdm
from src.core.util.data import get_true_env
from src.tianshou.tianshou.data import Batch, VectorReplayBuffer


def get_args_offline(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--construction_method', type=str, default='normal') # in {'normal', 'counterfactual', 'convolution'}
    parser.add_argument("--convolution_slice_num", type=int, default=10)
    parser.add_argument("--offline_repeat_num", type=int, default=10)

    args_new = parser.parse_known_args()[0]
    args.__dict__.update(args_new.__dict__)

    return args

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

    if args.env == 'MovieLensEnv-v0':  # need to add missing users and items in df_train
        assert hasattr(env, "lbe_user")
        assert hasattr(env, "lbe_item")

        df_user_num_mapped = df_user_num.copy()
        df_user_num_mapped.index = env.lbe_user.transform(df_user_num.index)

        df_train_part = df_train[["user_id", "item_id", args.yfeat]]
        df_train_part.loc[:, "user_id"] = env.lbe_user.transform(df_train_part["user_id"])
        df_train_part.loc[:, "item_id"] = env.lbe_item.transform(df_train_part["item_id"])

    elif args.env == 'KuaiEnv-v0':  # need to remove irrelated users and items in df_train
        assert hasattr(env, "lbe_user")
        df_user_num_mapped = df_user_num.loc[env.lbe_user.classes_]
        df_user_num_mapped = df_user_num_mapped.reset_index(drop=True)
        assert len(env.mat) == len(df_user_num_mapped)

        assert hasattr(env, "lbe_item")
        df_numpy = df_train[["user_id", "item_id", args.yfeat]].to_numpy()

        func = np.vectorize(lambda x: x in env.lbe_item.classes_)
        indices_valid_item = func(df_numpy[:, 1].astype(int))

        df_train_part = df_train[["user_id", "item_id", args.yfeat]].loc[indices_valid_item]
        # df_train_part["user_id"] = dummy_user = 0  # set to dummy user. Since these users are not in the evaluational environment.
        df_train_part = df_train_part.reset_index(drop=True)
        # df_user_items = df_train_part.groupby("user_id").agg(list)

        df_train_part["item_id"] = env.lbe_item.transform(df_train_part["item_id"])

        func_replace_user = np.vectorize(
            lambda x: x if x in env.lbe_user.classes_ else np.random.choice(env.lbe_user.classes_))
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
    Max_Min_Scaler = lambda x: (x - MIN) / (MAX - MIN)
    df_train_part.loc[:, args.yfeat] = df_train_part[args.yfeat].apply(Max_Min_Scaler)

    bins_ind, max_size, buffer_size = evenly_distribute_trajectories_to_bins(df_user_num_mapped, num_bins)

    if args.construction_method == 'normal':
        buffer_size_final = buffer_size
    elif args.construction_method == 'counterfactual':
        buffer_size_final = buffer_size * args.offline_repeat_num
    elif args.construction_method == 'convolution':
        buffer_size_final = buffer_size * args.convolution_slice_num
    buffer = VectorReplayBuffer(buffer_size_final, num_bins)


    # env.max_turn = max_size

    # df_user_items = df_train_part[["user_id", "item_id", args.yfeat]].groupby("user_id").agg(list)

    df_user_items = df_train_part.groupby("user_id").agg(list)
    for indices, users in tqdm(bins_ind.items(), total=len(bins_ind), desc="preparing offline data into buffer..."):
        for user in users:
            (user_reset, item_reset), info = env.reset()
            env.cur_user = user

            full_items = df_user_items.loc[user]["item_id"] 
            full_rewards = df_user_items.loc[user][args.yfeat]
            for head in range(0, len(full_items), math.ceil(len(full_items)/args.convolution_slice_num)):
                items = full_items[head:]
                rewards = full_rewards[head:]
                if args.construction_method == 'counterfactual':
                    item_reward_list = list(zip(items, rewards))
                    item_reward_numpy = np.array(item_reward_list).repeat(args.offline_repeat_num, axis=0)
                    item_reward_repeat = np.random.permutation(item_reward_numpy)
                    items = item_reward_repeat[:, 0].astype(int)
                    rewards = item_reward_repeat[:, 1].tolist()

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

                if(args.construction_method != 'convolution'): 
                    break

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

    return env, dataset, kwargs_um, buffer
