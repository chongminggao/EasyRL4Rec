import functools
import random
import sys
import traceback

import logzero
import numpy as np
import torch
from torch import nn

sys.path.extend([".", "./src", "./src/DeepCTR-Torch"])

from src.core.evaluation.evaluator_static import test_static_model_in_RL_env
from src.core.evaluation.loggers import LoggerEval_UserModel
from src.core.util.data import get_env_args, get_true_env
from src.core.userModel.user_model_ensemble import EnsembleModel
from src.core.evaluation.metrics import get_ranking_results

from src.core.util.loss import loss_pointwise_negative, loss_pointwise, loss_pairwise, loss_pairwise_pointwise
from usermodel_utils import get_datapath, prepare_dir_log, load_dataset_train, load_dataset_val, get_task, get_args_all, \
    get_args_dataset_specific

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def prepare_dataset(args, dataset, MODEL_SAVE_PATH, DATAPATH):
    dataset_train, df_user, df_item, x_columns, y_columns, ab_columns = \
        load_dataset_train(args, dataset, args.tau, args.entity_dim, args.feature_dim, MODEL_SAVE_PATH, DATAPATH)
    if not args.is_ab:
        ab_columns = None

    dataset_val, df_user_val, df_item_val = load_dataset_val(args, dataset, args.entity_dim, args.feature_dim)
    
    assert dataset_train.x_columns[1].vocabulary_size >= dataset_val.x_columns[1].vocabulary_size  # item_ids of training set should cover the test set!

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
        loss_fun = loss_pairwise
    if args.loss == "point":
        loss_fun = loss_pointwise
    if args.loss == "pointneg":
        loss_fun = loss_pointwise_negative
    if args.loss == "pointpair" or args.loss == "pairpoint" or args.loss == "pp":
        loss_fun = loss_pairwise_pointwise

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

def main(args, is_save=True):
    # %% 1. Prepare dir
    DATAPATH = get_datapath(args.env)
    args = get_env_args(args)
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare dataset
    env, dataset, kwargs_um = get_true_env(args, read_user_num=None)

    dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns = \
        prepare_dataset(args, dataset, MODEL_SAVE_PATH, DATAPATH)  # TODO 修改简化！

    # %% 3. Setup model
    task, task_logit_dim, is_ranking = get_task(args.env, args.yfeat)
    ensemble_models = setup_user_model(args, x_columns, y_columns, ab_columns,
                                        task, task_logit_dim, is_ranking, MODEL_SAVE_PATH)

    item_feat_domination = dataset.get_domination()
    ensemble_models.compile_RL_test(
        functools.partial(test_static_model_in_RL_env, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=args.is_ucb, need_transform=args.need_transform,
                          num_trajectory=args.num_trajectory, item_feat_domination=item_feat_domination,
                          force_length=args.force_length, top_rate=args.top_rate))

    # %% 5. Learn and evaluate model

    ensemble_models.fit_data(dataset_train, dataset_val,
                            batch_size=args.batch_size, epochs=args.epoch, shuffle=True,
                            callbacks=[LoggerEval_UserModel()])

    # %% 6. Save model
    if is_save:
        ensemble_models.save_all_models(dataset_val, x_columns, y_columns, df_user, df_item, df_user_val, df_item_val,
                                        dataset, args.is_userinfo, args.deterministic, args.use_auxiliary)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_args_dataset_specific(args_all.env)
    args_all.__dict__.update(args.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
