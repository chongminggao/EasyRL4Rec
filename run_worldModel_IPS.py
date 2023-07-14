# -*- coding: utf-8 -*-
# @Time    : 2022/9/30 20:23
# @Author  : Chongming GAO
# @FileName: run_worldModel.py
import argparse
import collections
import functools
import os
import random
import sys
import traceback

import logzero
import numpy as np
import pandas as pd
import torch
from torch import nn

from run_worldModel_ensemble import get_xy_columns, load_dataset_val, get_datapath, prepare_dir_log, get_task, \
    get_args_all, get_args_dataset_specific

sys.path.extend(["./src", "./src/DeepCTR-Torch"])
from core.evaluation.evaluator import test_static_model_in_RL_env
from core.configs import get_training_data, get_common_args, get_features, get_true_env, \
    get_training_item_domination
from core.user_model_ensemble import EnsembleModel
from core.evaluation.metrics import get_ranking_results
from core.static_dataset import StaticDataset
from core.util import negative_sampling

from util.utils import LoggerCallback_Update



def get_args_ips():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_model_name", type=str, default="DeepFM-IPS")
    parser.add_argument('--n_models', default=1, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument("--message", type=str, default="DeepFM-IPS")
    args = parser.parse_known_args()[0]
    return args


def load_dataset_train(args, user_features, item_features, reward_features, tau, entity_dim, feature_dim,
                       MODEL_SAVE_PATH, DATAPATH):
    df_train, df_user, df_item, list_feat = get_training_data(args.env)

    assert user_features[0] == "user_id"
    assert item_features[0] == "item_id"
    df_user = df_user[user_features[1:]]
    df_item = df_item[item_features[1:]]

    x_columns, y_columns, ab_columns = get_xy_columns(args, df_train, df_user, df_item, user_features, item_features,
                                                      entity_dim, feature_dim)

    neg_in_train = True if args.env == "KuaiRand-v0" and reward_features[0] != "watch_ratio_normed" else False
    neg_in_train = False  # todo: test for kuairand

    df_pos, df_neg = negative_sampling(df_train, df_item, df_user, reward_features[0],
                                       is_rand=True, neg_in_train=neg_in_train, neg_K=args.neg_K)

    df_x = df_pos[user_features + item_features]
    if reward_features[0] == "hybrid":  # for kuairand
        a = df_pos["long_view"] + df_pos["is_like"] + df_pos["is_click"]
        df_y = a > 0
        df_y = pd.DataFrame(df_y, dtype=int, columns=["hybrid"])
        df_pos["hybrid"] = df_y["hybrid"]
    else:
        df_y = df_pos[reward_features]

    df_x_neg = df_neg[user_features + item_features]
    df_x_neg = df_x_neg.rename(columns={k: k + "_neg" for k in df_x_neg.columns.to_numpy()})

    df_x_all = pd.concat([df_x, df_x_neg], axis=1)

    def compute_IPS(df_x_all, df_train) -> np.ndarray:
        IPS_item = collections.Counter(df_train['item_id'])
        IPS_data = df_x_all['item_id'].map(lambda x: IPS_item[x])
        IPS_data[IPS_data < 1] = 1
        IPS_data = 1.0 / IPS_data
        IPS_data_np = IPS_data.to_frame().to_numpy()
        return IPS_data_np

    IPS_data = compute_IPS(df_x_all, df_train)

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, IPS_data)

    return dataset, df_user, df_item, x_columns, y_columns, ab_columns

def prepare_dataset(args, user_features, item_features, reward_features, MODEL_SAVE_PATH, DATAPATH):
    dataset_train, df_user, df_item, x_columns, y_columns, ab_columns = \
        load_dataset_train(args, user_features, item_features, reward_features,
                           args.tau, args.entity_dim, args.feature_dim, MODEL_SAVE_PATH, DATAPATH)
    if not args.is_ab:
        ab_columns = None

    dataset_val, df_user_val, df_item_val = load_dataset_val(args, user_features, item_features, reward_features,
                                                             args.entity_dim, args.feature_dim)
    return dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns


def setup_world_model(args, x_columns, y_columns, ab_columns, task, task_logit_dim, is_ranking, MODEL_SAVE_PATH):
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)

    ensemble_models = EnsembleModel(args.n_models, args.message, MODEL_SAVE_PATH, x_columns, y_columns, task,
                                    task_logit_dim,
                                    dnn_hidden_units=args.dnn, dnn_hidden_units_var=args.dnn_var,
                                    seed=args.seed, l2_reg_dnn=args.l2_reg_dnn,
                                    device=device, ab_columns=ab_columns,
                                    dnn_activation=args.dnn_activation, init_std=0.001)

    if args.loss == "pair":
        loss_fun = loss_pairwise_IPS
    if args.loss == "point":
        loss_fun = loss_pointwise_IPS
    if args.loss == "pointneg":
        loss_fun = loss_pointwise_negative_IPS
    if args.loss == "pointpair" or args.loss == "pairpoint" or args.loss == "pp":
        loss_fun = loss_pairwise_pointwise_IPS

    ensemble_models.compile(optimizer=args.optimizer,
                            # loss_dict=task_loss_dict,
                            loss_func=functools.partial(loss_fun, args=args),
                            metric_fun={
                                "MAE": lambda y, y_predict: nn.functional.l1_loss(torch.from_numpy(y).type(torch.float),
                                                                                  torch.from_numpy(y_predict)).numpy(),
                                "MSE": lambda y, y_predict: nn.functional.mse_loss(
                                    torch.from_numpy(y).type(torch.float),
                                    torch.from_numpy(y_predict)).numpy(),
                                "RMSE": lambda y, y_predict: nn.functional.mse_loss(
                                    torch.from_numpy(y).type(torch.float),
                                    torch.from_numpy(
                                        y_predict)).numpy() ** 0.5
                            },
                            metric_fun_ranking=
                            functools.partial(get_ranking_results, K=args.rankingK,
                                              metrics=["Recall", "Precision", "NDCG", "HT", "MAP", "MRR"]
                                              ) if is_ranking else None,
                            metrics=None)

    # No evaluation step at offline stage
    # model.compile_RL_test(
    #     functools.partial(test_kuaishou, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
    #                       epsilon=args.epsilon, is_ucb=args.is_ucb))

    return ensemble_models




sigmoid = nn.Sigmoid()
def loss_pointwise_negative_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):


    loss_y = (((y_deepfm_pos - y) ** 2) * score).sum()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2)).sum()

    loss = loss_y + loss_y_neg
    return loss


def loss_pointwise_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):

    loss_y = (((y_deepfm_pos - y) ** 2) * score).sum()

    loss = loss_y

    return loss


def loss_pairwise_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log()*score).sum()
    loss = bpr_click

    return loss

def loss_pairwise_pointwise_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):

    loss_y = (((y_deepfm_pos - y) ** 2) * score).sum()
    bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log() * score).sum()
    loss = loss_y + args.bpr_weight * bpr_click
    return loss

def main(args):
    # %% 1. Prepare dir
    DATAPATH = get_datapath(args.env)
    args = get_common_args(args)
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare dataset
    user_features, item_features, reward_features = get_features(args.env, args.is_userinfo)

    dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns = \
        prepare_dataset(args, user_features, item_features, reward_features, MODEL_SAVE_PATH, DATAPATH)

    # %% 3. Setup model
    task, task_logit_dim, is_ranking = get_task(args.env, args.yfeat)
    ensemble_models = setup_world_model(args, x_columns, y_columns, ab_columns,
                                        task, task_logit_dim, is_ranking, MODEL_SAVE_PATH)

    env, env_task_class, kwargs_um = get_true_env(args, read_user_num=None)

    item_feat_domination = get_training_item_domination(args.env)
    ensemble_models.compile_RL_test(
        functools.partial(test_static_model_in_RL_env, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=args.is_ucb, need_transform=args.need_transform,
                          num_trajectory=args.num_trajectory, item_feat_domination=item_feat_domination,
                          force_length=args.force_length, top_rate=args.top_rate))

    # %% 5. Learn and evaluate model

    history_list = ensemble_models.fit_data(dataset_train, dataset_val,
                                            batch_size=args.batch_size, epochs=args.epoch, shuffle=True,
                                            callbacks=[LoggerCallback_Update(logger_path)])

    # %% 6. Save model
    # ensemble_models.get_save_entropy_mat(args.env, args.entropy_window)
    ensemble_models.save_all_models(dataset_val, x_columns, y_columns, df_user, df_item, df_user_val, df_item_val,
                                    user_features, item_features, args.deterministic)




if __name__ == '__main__':
    args_all = get_args_all()
    args = get_args_dataset_specific(args_all.env)
    args_ips = get_args_ips()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_ips.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
