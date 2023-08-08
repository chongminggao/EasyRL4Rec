import argparse
import functools
import random
import sys
import traceback

import logzero
import numpy as np
import torch
from torch import nn

sys.path.extend([".", "./src", "./src/DeepCTR-Torch"])

from core.evaluation.evaluator import test_static_model_in_RL_env
from core.util.data import get_common_args, get_features, get_true_env, \
    get_training_item_domination
from core.userModel.user_model_ensemble import EnsembleModel
from core.evaluation.metrics import get_ranking_results

from util.utils import LoggerCallback_Update
from core.util.loss import loss_pointwise_negative_Standard, loss_pointwise_Standard, loss_pairwise_Standard, loss_pairwise_pointwise_Standard
from usermodel_utils import get_datapath, prepare_dir_log, load_dataset_train, load_dataset_val, get_task, get_args_all, \
    get_args_dataset_specific


def get_args_epsilonGreedy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_model_name", type=str, default="EpsilonGreedy")
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--n_models', default=1, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument("--message", type=str, default="epsilon-greedy")
    args = parser.parse_known_args()[0]
    return args


def prepare_dataset(args, user_features, item_features, reward_features, MODEL_SAVE_PATH, DATAPATH):
    dataset_train, df_user, df_item, x_columns, y_columns, ab_columns = \
        load_dataset_train(args, user_features, item_features, reward_features,
                           args.tau, args.entity_dim, args.feature_dim, MODEL_SAVE_PATH, DATAPATH)
    if not args.is_ab:
        ab_columns = None

    dataset_val, df_user_val, df_item_val = load_dataset_val(args, user_features, item_features, reward_features,
                                                             args.entity_dim, args.feature_dim)
    return dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns


def setup_user_model(args, x_columns, y_columns, ab_columns, task, task_logit_dim, is_ranking, MODEL_SAVE_PATH):
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
        loss_fun = loss_pairwise_Standard
    if args.loss == "point":
        loss_fun = loss_pointwise_Standard
    if args.loss == "pointneg":
        loss_fun = loss_pointwise_negative_Standard
    if args.loss == "pointpair" or args.loss == "pairpoint" or args.loss == "pp":
        loss_fun = loss_pairwise_pointwise_Standard

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

    return ensemble_models


def main(args, is_save=False):
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
    ensemble_models = setup_user_model(args, x_columns, y_columns, ab_columns,
                                        task, task_logit_dim, is_ranking, MODEL_SAVE_PATH)

    env, env_task_class, kwargs_um = get_true_env(args, read_user_num=None)

    item_feat_domination = get_training_item_domination(args.env)
    ensemble_models.compile_RL_test(
        functools.partial(test_static_model_in_RL_env, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=args.is_ucb, need_transform=args.need_transform,
                          num_trajectory=args.num_trajectory, item_feat_domination=item_feat_domination,
                          force_length=args.force_length, top_rate=args.top_rate, draw_bar=args.draw_bar))

    # %% 5. Learn and evaluate model

    history_list = ensemble_models.fit_data(dataset_train, dataset_val,
                                            batch_size=args.batch_size, epochs=args.epoch, shuffle=True,
                                            callbacks=[LoggerCallback_Update(logger_path)])

    # %% 6. Save model
    if is_save:
        ensemble_models.get_save_entropy_mat(args.env, args.entropy_window)
        ensemble_models.save_all_models(dataset_val, x_columns, y_columns, df_user, df_item, df_user_val, df_item_val,
                                        user_features, item_features, args.deterministic)



if __name__ == '__main__':
    args_all = get_args_all()
    args = get_args_dataset_specific(args_all.env)
    args_epsilon = get_args_epsilonGreedy()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_epsilon.__dict__)

    try:
        main(args_all, is_save=False)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
