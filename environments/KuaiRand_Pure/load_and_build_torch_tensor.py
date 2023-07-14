# -*- coding: utf-8 -*-
# @Time    : 2022/9/18 12:10
# @Author  : Chongming GAO
# @FileName: load_and_transfer.py

import os
from collections import namedtuple, OrderedDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "data")


def input_from_feature_columns(X, feature_columns, embedding_dict, feature_index, support_dense: bool, device):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError(
            "DenseFeat is not supported in dnn_feature_columns")

    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
        feat in sparse_feature_columns]

    dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                        dense_feature_columns]

    return sparse_embedding_list, dense_value_list


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse,
                                           padding_idx=feat.padding_idx)
         for feat in
         sparse_feature_columns}
    )

    for tensor in embedding_dict.values():
        if tensor.padding_idx is None:
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        else:
            nn.init.normal_(tensor.weight[:tensor.padding_idx], mean=0, std=init_std)
            nn.init.normal_(tensor.weight[tensor.padding_idx + 1:], mean=0, std=init_std)

    return embedding_dict.to(device)


def build_input_features(feature_columns):
    # Return OrderedDict: {feature_name:(start, start+dimension)}

    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name="default_group"):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class SparseFeatP(SparseFeat):
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name="default_group", padding_idx=None):
        return super(SparseFeatP, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                               embedding_name, group_name)

    def __init__(self, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                 group_name="default_group", padding_idx=None):
        self.padding_idx = padding_idx


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


class StaticDataset(Dataset):
    def __init__(self, x_columns, y_columns, num_workers=4):
        self.x_columns = x_columns
        self.y_columns = y_columns

        self.num_workers = num_workers

        self.len = 0
        self.neg_items_info = None
        self.ground_truth = None

        self.all_item_ranking = False

    def compile_dataset(self, df_x, df_y, score=None):
        self.x_numpy = df_x.to_numpy()
        self.y_numpy = df_y.to_numpy()

        if score is None:
            self.score = np.zeros([len(self.x_numpy), 1])
        else:
            self.score = score

        self.len = len(self.x_numpy)

    def get_dataset_train(self):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.x_numpy),
                                                 torch.from_numpy(self.y_numpy),
                                                 torch.from_numpy(self.score))
        return dataset

    def get_dataset_eval(self):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.x_numpy),
                                                 torch.from_numpy(self.y_numpy))
        return dataset

    def get_y(self):
        return self.y_numpy

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        x = self.x_numpy[index]
        y = self.y_numpy[index]
        return x, y


def load_user_info():
    print("load user features")
    filepath = os.path.join(DATAPATH, 'user_features_pure.csv')
    df_user = pd.read_csv(filepath, usecols=['user_id', 'user_active_degree',
                                             'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                                             'fans_user_num_range', 'friend_user_num_range',
                                             'register_days_range'] + [f'onehot_feat{x}' for x in range(18)]
                          )
    for col in ['user_active_degree',
                'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                'fans_user_num_range', 'friend_user_num_range', 'register_days_range']:

        df_user[col] = df_user[col].map(lambda x: chr(0) if x == 'UNKNOWN' else x)
        lbe = LabelEncoder()
        df_user[col] = lbe.fit_transform(df_user[col])
        # print(lbe.classes_)
        if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
            assert lbe.classes_[0] in {-124, chr(0)}
            # do not add one
        else:
            df_user[col] += 1
    for col in [f'onehot_feat{x}' for x in range(18)]:
        df_user.loc[df_user[col].isna(), col] = -124
        lbe = LabelEncoder()
        df_user[col] = lbe.fit_transform(df_user[col])
        # print(lbe.classes_)
        if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
            assert lbe.classes_[0] in {-124, chr(0)}
            # do not add one
        else:
            df_user[col] += 1

    df_user = df_user.set_index("user_id")
    return df_user


def load_category():
    print("load item feature")
    filepath = os.path.join(DATAPATH, 'video_features_basic_pure.csv')
    df_item = pd.read_csv(filepath, usecols=["tag"], dtype=str)
    ind = df_item['tag'].isna()
    df_item['tag'].loc[~ind] = df_item['tag'].loc[~ind].map(lambda x: eval(f"[{x}]"))
    df_item['tag'].loc[ind] = [[-1]] * ind.sum()

    list_feat = df_item['tag'].to_list()

    df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2'], dtype=int)
    df_feat.index.name = "video_id"
    df_feat[df_feat.isna()] = -1
    df_feat = df_feat + 1
    df_feat = df_feat.astype(int)

    return list_feat, df_feat


def get_df_kuairand(name, is_sort=True):
    filename = os.path.join(DATAPATH, name)
    df_data = pd.read_csv(filename,
                          usecols=['user_id', 'video_id', 'time_ms', 'is_like', 'is_click', 'long_view',
                                   'play_time_ms', 'duration_ms'])

    df_data['watch_ratio'] = df_data["play_time_ms"] / df_data["duration_ms"]
    df_data.loc[df_data['watch_ratio'].isin([np.inf, np.nan]), 'watch_ratio'] = 0
    df_data.loc[df_data['watch_ratio'] > 5, 'watch_ratio'] = 5
    df_data['duration_ms'] /= 1e5
    df_data.rename(columns={"time_ms": "timestamp"}, inplace=True)
    df_data["timestamp"] /= 1e3

    # load feature info
    list_feat, df_feat = load_category()
    df_data = df_data.join(df_feat, on=['video_id'], how="left")

    # load user info
    df_user = load_user_info()
    df_data = df_data.join(df_user, on=['user_id'], how="left")

    # get user sequences
    if is_sort:
        df_data.sort_values(["user_id", "timestamp"], inplace=True)
        df_data.reset_index(drop=True, inplace=True)

    return df_data, df_user, df_feat


def load_dataset_kuairand(user_features, item_features, reward_features, entity_dim, feature_dim):
    df_train, df_user, df_feat = get_df_kuairand("log_standard_4_08_to_4_21_pure.csv")

    df_x = df_train[user_features + item_features]
    if reward_features[0] == "hybrid":
        a = df_train["long_view"] + df_train["is_like"] + df_train["is_click"]
        df_y = a > 0
        df_y = pd.DataFrame(df_y, dtype=int, columns=["hybrid"])
    else:
        df_y = df_train[reward_features]
    print(f"Train: {reward_features}: {df_y.sum()[0]}/{len(df_y)}")

    x_columns = [SparseFeatP("user_id", df_train['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim, padding_idx=0) for col in
                 user_features[1:]] + \
                [SparseFeatP("video_id", df_train['video_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("feat{}".format(i),
                             df_feat.max().max() + 1,
                             embedding_dim=feature_dim,
                             embedding_name="feat",  # Share the same feature!
                             padding_idx=0  # using padding_idx in embedding!
                             ) for i in range(3)] + \
                [DenseFeat("duration_ms", 1)]

    y_columns = [DenseFeat("y", 1)]

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x, df_y)

    return dataset, x_columns, y_columns


if __name__ == '__main__':
    user_features = ["user_id", 'user_active_degree', 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                     'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] \
                    + [f'onehot_feat{x}' for x in range(18)]
    item_features = ["video_id"] + ["feat" + str(i) for i in range(3)] + ["duration_ms"]
    reward_features = ["is_click"]
    # 这里的reward_features就是label，这里是is_click，可以换成其他feedback信号。

    entity_dim = 8
    feature_dim = 8

    dataset_train, x_columns, y_columns = load_dataset_kuairand(user_features, item_features, reward_features,
                                                                entity_dim,
                                                                feature_dim)

    embedding_dict = create_embedding_matrix(x_columns)

    print(embedding_dict)

    # 每一维度是什么特征。
    feature_index = build_input_features(x_columns)
    print(feature_index)

    train_loader = DataLoader(
        dataset=dataset_train.get_dataset_train(), shuffle=True, batch_size=2048,
        num_workers=dataset_train.num_workers)

    # 这里的myscore是其他用途，不用管。y就是label，即上面设置的is_click。
    for i, (x, y, myscore) in enumerate(train_loader):
        sparse_embedding_list, dense_value_list = input_from_feature_columns(x, x_columns,
                                                                             embedding_dict,
                                                                             feature_index=feature_index,
                                                                             support_dense=True, device='cpu')
        representation = combined_dnn_input(sparse_embedding_list, dense_value_list)
        # 这里已经转化为了torch.tensor。后面就用这个去输入model就行了。

    # 可视化
    print(representation)

