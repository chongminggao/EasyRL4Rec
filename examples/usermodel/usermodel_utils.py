import os
import argparse
import collections
import json
import time
import datetime
import logzero
import numpy as np
import pandas as pd

from core.util.inputs import SparseFeatP
from deepctr_torch.inputs import DenseFeat
from core.util.static_dataset import StaticDataset
from core.util.utils import negative_sampling, create_dir

from environments.KuaiRec.KuaiData import compute_exposure_effect_kuaiRec

CODEPATH = os.path.dirname(__file__)

def get_args_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument("--bpr_weight", type=float, default=0.5)
    parser.add_argument('--neg_K', default=5, type=int)
    parser.add_argument('--n_models', default=5, type=int)

    # recommendation related:
    # parser.add_argument('--not_softmax', action="store_false")
    parser.add_argument('--is_softmax', dest='is_softmax', action='store_true')
    parser.add_argument('--no_softmax', dest='is_softmax', action='store_false')
    parser.set_defaults(is_softmax=False)
    parser.add_argument("--num_trajectory", type=int, default=200)
    parser.add_argument("--force_length", type=int, default=10)
    parser.add_argument("--top_rate", type=float, default=0.8)

    parser.add_argument('--is_deterministic', dest='deterministic', action='store_true')
    parser.add_argument('--no_deterministic', dest='deterministic', action='store_false')
    parser.set_defaults(deterministic=True)

    parser.add_argument('--is_draw_bar', dest='draw_bar', action='store_true')
    parser.add_argument('--no_draw_bar', dest='draw_bar', action='store_false')
    parser.set_defaults(draw_bar=False)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    parser.add_argument("--loss", type=str, default='pointneg') # in {"pointneg", "point", "pair", "pp"}
    parser.add_argument('--rankingK', default=(20, 10, 5), type=int, nargs="+")
    parser.add_argument('--max_turn', default=30, type=int)

    parser.add_argument('--l2_reg_dnn', default=0.1, type=float)
    parser.add_argument('--lambda_ab', default=10, type=float)

    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument('--is_ucb', dest='is_ucb', action='store_true')
    parser.add_argument('--no_ucb', dest='is_ucb', action='store_false')
    parser.set_defaults(is_ucb=False)

    parser.add_argument("--dnn_activation", type=str, default="relu")
    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--entity_dim", type=int, default=8)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--dnn', default=(128, 128), type=int, nargs="+")
    parser.add_argument('--dnn_var', default=(), type=int, nargs="+")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--cuda', default=0, type=int)

    # exposure parameters:
    parser.add_argument('--tau', default=0, type=float)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=False)
    parser.add_argument("--message", type=str, default="UM")

    args = parser.parse_known_args()[0]
    return args


def get_args_dataset_specific(envname):

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_random_init', dest='random_init', action='store_true')
    parser.add_argument('--no_random_init', dest='random_init', action='store_false')
    parser.set_defaults(random_init=True)

    if envname == 'CoatEnv-v0':
        parser.add_argument("--feature_dim", type=int, default=8)
        parser.add_argument("--entity_dim", type=int, default=8)
        parser.add_argument('--batch_size', default=1024, type=int)
        # parser.add_argument("--dnn_activation", type=str, default="prelu")
        parser.add_argument('--leave_threshold', default=10, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
    elif envname == 'YahooEnv-v0':
        parser.add_argument("--feature_dim", type=int, default=8)
        parser.add_argument("--entity_dim", type=int, default=8)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--leave_threshold', default=120, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
    elif envname == 'MovieLensEnv-v0':
        parser.add_argument("--feature_dim", type=int, default=8)
        parser.add_argument("--entity_dim", type=int, default=8)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--leave_threshold', default=120, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
    elif envname == 'KuaiEnv-v0':
        parser.add_argument('--neg_K', default=3, type=int)
        parser.add_argument("--feature_dim", type=int, default=8)
        parser.add_argument("--entity_dim", type=int, default=8)
        parser.add_argument('--batch_size', default=4096, type=int)
        # parser.add_argument("--dnn_activation", type=str, default="swish")
        parser.add_argument('--leave_threshold', default=0, type=int)  # todo
        parser.add_argument('--num_leave_compute', default=1, type=int)  # todo
    elif envname == 'KuaiRand-v0':
        parser.add_argument("--yfeat", type=str, default='is_click')
        parser.add_argument("--feature_dim", type=int, default=4)
        parser.add_argument("--entity_dim", type=int, default=4)
        parser.add_argument('--batch_size', default=4096, type=int)
        parser.add_argument('--leave_threshold', default=10, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
    else:
        raise (
            "envname should be in the following four datasets: {'CoatEnv-v0', 'YahooEnv-v0', 'KuaiEnv-v0', 'KuaiRand-v0'}")

    args = parser.parse_known_args()[0]
    return args


def get_datapath(envname):
    DATAPATH = None
    if envname == 'CoatEnv-v0':
        DATAPATH = os.path.join(CODEPATH, "environments", "coat")
    elif envname == 'YahooEnv-v0':
        DATAPATH = os.path.join(CODEPATH, "environments", "YahooR3")
    elif envname == 'MovieLensEnv-v0':
        DATAPATH = os.path.join(CODEPATH, "environments", "MovieLens")
    elif envname == 'KuaiEnv-v0':
        DATAPATH = os.path.join(CODEPATH, "environments", "KuaiRec", "data")
    elif envname == 'KuaiRand-v0':
        DATAPATH = os.path.join(CODEPATH, "environments", "KuaiRand_Pure", "data")
    return DATAPATH


def prepare_dir_log(args):
    args.entity_dim = args.feature_dim
    # %% 1. Create dirs
    MODEL_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.user_model_name)

    create_dirs = [os.path.join(".", "saved_models"),
                   os.path.join(".", "saved_models", args.env),
                   MODEL_SAVE_PATH,
                   os.path.join(MODEL_SAVE_PATH, "logs"),
                   os.path.join(MODEL_SAVE_PATH, "matsPre"),
                   os.path.join(MODEL_SAVE_PATH, "matsVar"),
                   os.path.join(MODEL_SAVE_PATH, "entropy"),
                   os.path.join(MODEL_SAVE_PATH, "embeddings"),
                   os.path.join(MODEL_SAVE_PATH, "params"),
                   os.path.join(MODEL_SAVE_PATH, "models")]
    create_dir(create_dirs)

    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)
    logzero.logger.info(json.dumps(vars(args), indent=2))
    return MODEL_SAVE_PATH, logger_path


def load_dataset_train(args, dataset, tau, entity_dim, feature_dim,
                       MODEL_SAVE_PATH, DATAPATH):
    user_features, item_features, reward_features = dataset.get_features(args.is_userinfo)
    df_train, df_user, df_item, list_feat = dataset.get_train_data()

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

    if tau == 0:
        exposure_pos = np.zeros([len(df_x_all), 1])
    else:
        timestamp = df_pos['timestamp']
        exposure_pos = compute_exposure_effect_kuaiRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH)

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, exposure_pos)

    return dataset, df_user, df_item, x_columns, y_columns, ab_columns

def load_dataset_train_IPS(args, dataset, tau, entity_dim, feature_dim,
                       MODEL_SAVE_PATH, DATAPATH):
    user_features, item_features, reward_features = dataset.get_features(args.is_userinfo)
    df_train, df_user, df_item, list_feat = dataset.get_train_data()

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

def load_dataset_val(args, dataset, entity_dim, feature_dim):
    user_features, item_features, reward_features = dataset.get_features(args.is_userinfo)
    df_val, df_user_val, df_item_val, list_feat = dataset.get_val_data()

    assert user_features[0] == "user_id"
    assert item_features[0] == "item_id"
    df_user_val = df_user_val[user_features[1:]]
    df_item_val = df_item_val[item_features[1:]]

    df_x = df_val[user_features + item_features]
    if reward_features[0] == "hybrid":  # for kuairand
        a = df_val["long_view"] + df_val["is_like"] + df_val["is_click"]
        df_y = a > 0
        df_y = pd.DataFrame(df_y, dtype=int, columns=["hybrid"])
    else:
        df_y = df_val[reward_features]

    x_columns, y_columns, ab_columns = get_xy_columns(args, df_val, df_user_val, df_item_val, user_features,
                                                      item_features,
                                                      entity_dim, feature_dim)

    dataset_val = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_val.compile_dataset(df_x, df_y)

    dataset_val.set_df_item_val(df_item_val)
    dataset_val.set_df_user_val(df_user_val)

    assert dataset_val.x_columns[0].name == "user_id"
    dataset_val.set_user_col(0)
    assert dataset_val.x_columns[len(user_features)].name == "item_id"
    dataset_val.set_item_col(len(user_features))

    if not any(df_y.to_numpy() % 1):  # if df_y is int.
        # make sure the label is binary

        df_binary = pd.concat([df_val[["user_id", "item_id"]], df_y], axis=1)
        df_ones = df_binary.loc[df_binary[reward_features[0]] > 0]
        ground_truth = df_ones[["user_id", "item_id"] + reward_features].groupby("user_id").agg(list)
        ground_truth.rename(columns={"item_id": "item_id", reward_features[0]: "y"}, inplace=True)

        # for ranking purpose.
        threshold = args.rating_threshold
        index = ground_truth["y"].map(lambda x: [True if i >= threshold else False for i in x])
        df_temp = pd.DataFrame(index)
        df_temp.rename(columns={"y": "ind"}, inplace=True)
        df_temp["y"] = ground_truth["y"]
        df_temp["true_id"] = ground_truth["item_id"]
        df_true_id = df_temp.apply(lambda x: np.array(x["true_id"])[x["ind"]].tolist(), axis=1)
        df_true_y = df_temp.apply(lambda x: np.array(x["y"])[x["ind"]].tolist(), axis=1)

        if args.is_binarize:
            df_true_y = df_true_y.map(lambda x: [1] * len(x))

        ground_truth_revise = pd.concat([df_true_id, df_true_y], axis=1)
        ground_truth_revise.rename(columns={0: "item_id", 1: "y"}, inplace=True)
        dataset_val.set_ground_truth(ground_truth_revise)

        if args.all_item_ranking:
            # See: Walid Krichene, Steffen Rendle. On Sampled Metrics for Item Recommendation. KDD 2020 Best Paper.
            dataset_val.set_all_item_ranking_in_evaluation(args.all_item_ranking)

            df_x_complete = construct_complete_val_x(dataset_val, df_user_val, df_item_val, user_features,
                                                     item_features)
            df_y_complete = pd.DataFrame(np.zeros(len(df_x_complete)), columns=df_y.columns)

            dataset_complete = StaticDataset(x_columns, y_columns, num_workers=4)
            dataset_complete.compile_dataset(df_x_complete, df_y_complete)
            dataset_val.set_dataset_complete(dataset_complete)

    return dataset_val, df_user_val, df_item_val


def get_task(envname, yfeat):
    task = None
    task_logit_dim = 1
    if envname == 'CoatEnv-v0':
        task = "regression"
        is_ranking = True
    elif envname == 'YahooEnv-v0':
        task = "regression"
        is_ranking = True
    elif envname == 'MovieLensEnv-v0':
        task = "regression"
        is_ranking = True
    elif envname == 'KuaiEnv-v0':
        task = "regression"
        is_ranking = False
    elif envname == 'KuaiRand-v0':
        task = "regression" if yfeat == "watch_ratio_normed" else "binary"
        is_ranking = True
    return task, task_logit_dim, is_ranking


def get_xy_columns(args, df_data, df_user, df_item, user_features, item_features, entity_dim, feature_dim):
    if args.env == "KuaiRand-v0" or args.env == "KuaiEnv-v0":
        feat = [x for x in df_item.columns if x[:4] == "feat"]
        x_columns = [SparseFeatP("user_id", df_data['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim, padding_idx=0) for col in
                     user_features[1:]] + \
                    [SparseFeatP("item_id", df_data['item_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(x,
                                 df_item[feat].max().max() + 1,
                                 embedding_dim=feature_dim,
                                 embedding_name="feat",  # Share the same feature!
                                 padding_idx=0  # using padding_idx in embedding!
                                 ) for x in feat] + \
                    [DenseFeat("duration_normed", 1)]

    else:
        x_columns = [SparseFeatP("user_id", df_data['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim) for col in user_features[1:]] + \
                    [SparseFeatP("item_id", df_data['item_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_item[col].max() + 1, embedding_dim=feature_dim) for col in item_features[1:]]

    ab_columns = [SparseFeatP("alpha_u", df_data['user_id'].max() + 1, embedding_dim=1)] + \
                 [SparseFeatP("beta_i", df_data['item_id'].max() + 1, embedding_dim=1)]

    y_columns = [DenseFeat("y", 1)]
    return x_columns, y_columns, ab_columns

def construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features):
    user_ids = np.unique(dataset_val.x_numpy[:, dataset_val.user_col].astype(int))

    # user_ids = random.sample(user_ids.tolist(),100)
    user_ids = user_ids[:10000] # todo: for speeding up, we only use 10000 users for visual the bars.
    logzero.logger.info("#####################\nNote that we use only 10000 users for static evaluation!!\n#####################")
    item_ids = np.unique(dataset_val.x_numpy[:, dataset_val.item_col].astype(int))

    df_user_complete = pd.DataFrame(
        df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
        columns=df_user.reset_index()[user_features].columns)
    df_item_complete = pd.DataFrame(np.tile(df_item.loc[item_ids].reset_index()[item_features], (len(user_ids), 1)),
                                    columns=df_item.loc[item_ids].reset_index()[item_features].columns)

    df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)
    return df_x_complete