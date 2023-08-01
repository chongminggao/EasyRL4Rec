# -*- coding: utf-8 -*-
# @Time    : 2022/11/14 14:50
# @Author  : Chongming GAO
# @FileName: user_model_ensemble.py
import os
import pickle
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool, Process

import logzero
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.configs import get_training_data
from core.util.inputs import input_from_feature_columns
from core.userModel.static_dataset import StaticDataset
from core.userModel.user_model_pairwise_variance import UserModel_Pairwise_Variance
from logzero import logger

from deepctr_torch.inputs import combined_dnn_input, build_input_features


# def collect_res(res_list, res):


class EnsembleModel():
    def __init__(self, num_models, message, MODEL_SAVE_PATH, *args, **kwargs):

        self.user_models = [UserModel_Pairwise_Variance(*args, **kwargs) for i in range(num_models)]

        self.MODEL_SAVE_PATH = MODEL_SAVE_PATH
        self.PREDICTION_MAT_PATH = os.path.join(MODEL_SAVE_PATH, "matsPre", f"[{message}]_matPre.pickle")
        self.VAR_MAT_PATH = os.path.join(MODEL_SAVE_PATH, "matsVar", f"[{message}]_matVar.pickle")
        self.Entropy_PATH = os.path.join(MODEL_SAVE_PATH, "entropy")
        self.MODEL_PARAMS_PATH = os.path.join(MODEL_SAVE_PATH, "params", f"[{message}]_params.pickle")
        self.MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "models", f"[{message}]_model.pt")
        self.MODEL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings",
                                                 f"[{message}]_emb.pt")  # todo: deprecated
        self.USER_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{message}]_emb_user.pt")
        self.ITEM_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{message}]_emb_item.pt")
        self.USER_VAL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{message}]_emb_user_val.pt")
        self.ITEM_VAL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{message}]_emb_item_val.pt")

    def compile(self, *args, **kwargs):
        for model in self.user_models:
            model.compile(*args, **kwargs)

    def train(self, *args, **kwargs):
        for model in self.user_models:
            model.train(*args, **kwargs)
        return self.user_models

    def eval(self, *args, **kwargs):
        for model in self.user_models:
            model.eval(*args, **kwargs)
        return self.user_models

    def compile_RL_test(self, *args, **kwargs):
        for model in self.user_models:
            model.compile_RL_test(*args, **kwargs)

    def fit_data(self, *args, **kwargs):

        # pool = Pool()
        # for model in self.user_models:
        #     res = pool.apply_async(func=fit_data_handler, args=(model,) + args, kwds=kwargs, callback=lambda x: print(x))  # 实例化进程对象
        #     print(res.get())
        #
        # pool.close()
        # pool.join()

        history_list = []
        for model in self.user_models:
            history = model.fit_data(*args, **kwargs)
            # print(history)
            # logger.info(history.history)
            logger.info("\n")
            history_list.append(history.history)

        logger.info("============ Summarized results =============")
        for hist in history_list:
            logger.info(hist)
            # logger.info("\n")

        return history_list

    def load_all_models(self):
        for i, model in enumerate(self.user_models):
            MODEL_PATH_new = get_detailed_path(self.MODEL_PATH, i)

            model.load_state_dict(torch.load(MODEL_PATH_new))

            # todo: need to cuda??

            # model = model.cpu()
            # model.linear_model.device = "cpu"
            # model.linear.device = "cpu"
            #

    def load_val_user_item_embedding(self, model_i=0, freeze_emb=True):
        user_embedding = torch.load(get_detailed_path(self.USER_VAL_EMBEDDING_PATH, model_i))
        item_embedding = torch.load(get_detailed_path(self.ITEM_VAL_EMBEDDING_PATH, model_i))
        saved_embedding = torch.nn.ModuleDict(
            {"feat_user": torch.nn.Embedding.from_pretrained(user_embedding, freeze=freeze_emb),
             "feat_item": torch.nn.Embedding.from_pretrained(item_embedding, freeze=freeze_emb)})
        return saved_embedding

    def compute_mean_var(self, dataset_val, df_user, df_item, user_features, item_features, x_columns, y_columns):
        df_x_complete = construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features)
        n_user, n_item = df_x_complete[["user_id", "item_id"]].nunique()

        print("predict all users' rewards on all items")

        dataset_um = StaticDataset(x_columns, y_columns, num_workers=4)
        dataset_um.compile_dataset(df_x_complete, pd.DataFrame(np.zeros([len(df_x_complete), 1]), columns=["y"]))

        sample_num = len(dataset_um)
        batch_size = 10000
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        test_loader = DataLoader(dataset=dataset_um.get_dataset_eval(), shuffle=False, batch_size=batch_size,
                                 num_workers=dataset_um.num_workers)

        mean_mat_list, var_mat_list = [], []
        for model in self.user_models:
            mean_mat, var_mat = get_one_predicted_res(model, df_x_complete, test_loader, steps_per_epoch)
            mean_mat_list.append(mean_mat)
            var_mat_list.append(var_mat)

        return mean_mat_list, var_mat_list

    def get_prediction_and_maxvar(self, mean_mat_list, var_mat_list, deterministic):
        if deterministic:
            prediction = np.stack(mean_mat_list).mean(0)
        else:
            ind_mat = np.random.randint(np.ones(mean_mat_list[0].shape) * len(mean_mat_list))
            mean_tensor = np.stack(mean_mat_list)
            var_tensor = np.stack(var_mat_list)

            # mean_sampled = np.zeros_like(mean_mat_list[0])
            # var_sampled = np.zeros_like(var_mat_list[0])
            prediction = np.zeros_like(var_mat_list[0])

            for i in range(len(prediction)):
                for j in range(len(prediction[0])):
                    # mean_sampled[i][j] = mean_tensor[ind_mat[i][j],i,j]
                    # var_sampled[i][j] = var_tensor[ind_mat[i][j], i, j]
                    prediction[i][j] = mean_tensor[ind_mat[i][j], i, j] + \
                                       np.sqrt(var_tensor[ind_mat[i][j], i, j]) * np.random.normal()

        var_max = np.stack(var_mat_list).max(0)

        return prediction, var_max

    def get_save_entropy_mat(self, envname, entropy_window):
        df_train, df_user, df_item, list_feat = get_training_data(envname)

        num_item = df_train["item_id"].nunique()
        if not "timestamp" in df_train.columns:
            df_train.rename(columns={"time_ms": "timestamp"}, inplace=True)

        def get_entropy(mylist, need_count=True):
            if len(mylist) <= 1:
                return 1
            if need_count:
                cnt_dict = Counter(mylist)
            else:
                cnt_dict = mylist
            prob = np.array(list(cnt_dict.values())) / sum(cnt_dict.values())
            log_prob = np.log2(prob)
            entropy = - np.sum(log_prob * prob) / np.log2(len(cnt_dict))
            # entropy = - np.sum(log_prob * prob) / np.log2(len(cnt_dict) + 1)
            return entropy

        entropy_user, map_entropy = None, None

        # if 0 in entropy_window:
        #     df_train = df_train.sort_values("user_id")
        #     interaction_list = df_train[["user_id", "item_id"]].groupby("user_id").agg(list)
        #     entropy_user = interaction_list["item_id"].map(partial(get_entropy))
        #
        #     savepath = os.path.join(self.Entropy_PATH, "user_entropy.csv")
        #     entropy_user.to_csv(savepath, index=True)

        if len(set(entropy_window) - set([0])):

            df_uit = df_train[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"])

            map_hist_count = defaultdict(lambda: defaultdict(int))
            lastuser = int(-1)

            def update_map(map_hist_count, hist_tra, item, require_len):
                if len(hist_tra) < require_len:
                    return
                # if require_len == 0:
                #     map_hist_count[tuple()][item] += 1
                # else:
                map_hist_count[tuple(sorted(hist_tra[-require_len:]))][item] += 1

            hist_tra = []
            # for k, (user, item, time) in tqdm(df_uit.iterrows(), total=len(df_uit), desc="count frequency..."):
            for (user, item, time) in tqdm(df_uit.to_numpy(), total=len(df_uit), desc="count frequency..."):
                user = int(user)
                item = int(item)

                if user != lastuser:
                    lastuser = user
                    hist_tra = []

                for require_len in set(entropy_window) - set([0]):
                    update_map(map_hist_count, hist_tra, item, require_len)
                hist_tra.append(item)

            map_entropy = {}
            for k, v in tqdm(map_hist_count.items(), total=len(map_hist_count), desc="compute entropy..."):
                map_entropy[k] = get_entropy(v, need_count=False)

            savepath = os.path.join(self.Entropy_PATH, "map_entropy.pickle")
            pickle.dump(map_entropy, open(savepath, 'wb'))

            # print(map_hist_count)

        return entropy_user, map_entropy

    def save_all_models(self, dataset_val, x_columns, y_columns, df_user, df_item, df_user_val, df_item_val,
                        user_features, item_features, deterministic):

        # (1) Compute and save Mat
        mean_mat_list, var_mat_list = self.compute_mean_var(dataset_val, df_user, df_item, user_features, item_features,
                                                            x_columns, y_columns)

        prediction, var_max = self.get_prediction_and_maxvar(mean_mat_list, var_mat_list, deterministic)

        with open(self.PREDICTION_MAT_PATH, "wb") as f:
            pickle.dump(prediction, f)
        with open(self.VAR_MAT_PATH, "wb") as f:
            pickle.dump(var_max, f)

        # (2) Save params

        model_parameters = self.user_models[0].model_param
        model_parameters.update({"n_models": len(self.user_models),
                                 "device": "cpu"})
        with open(self.MODEL_PARAMS_PATH, "wb") as output_file:
            pickle.dump(model_parameters, output_file)

        # (3) Save Model
        #  To cpu

        # model = user_model.cpu()
        # model.linear_model.device = "cpu"
        # model.linear.device = "cpu"
        #
        # torch.save(model.state_dict(), MODEL_PATH)

        for i, model in enumerate(self.user_models):
            MODEL_PATH_new = get_detailed_path(self.MODEL_PATH, i)

            model = model.cpu()
            model.linear_model.device = "cpu"
            model.linear.device = "cpu"
            torch.save(model.state_dict(), MODEL_PATH_new)

        # (4) Save Embedding
        # torch.save(model.embedding_dict.state_dict(), MODEL_EMBEDDING_PATH)

        def save_embedding(model, df_save, columns, SAVEPATH):
            df_save = df_save.reset_index(drop=False)
            df_save = df_save[[column.name for column in columns]]

            feature_index = build_input_features(columns)
            tensor_save = torch.FloatTensor(df_save.to_numpy())
            sparse_embedding_list, dense_value_list = input_from_feature_columns(tensor_save, columns,
                                                                                 model.embedding_dict,
                                                                                 feature_index=feature_index,
                                                                                 support_dense=True, device='cpu')
            representation_save = combined_dnn_input(sparse_embedding_list, dense_value_list)
            torch.save(representation_save, SAVEPATH)
            return representation_save

        user_columns = x_columns[:len(user_features)]
        item_columns = x_columns[len(user_features):]

        for i, model in enumerate(self.user_models):
            ITEM_EMBEDDING_PATH_new = get_detailed_path(self.ITEM_EMBEDDING_PATH, i)
            USER_EMBEDDING_PATH_new = get_detailed_path(self.USER_EMBEDDING_PATH, i)
            ITEM_VAL_EMBEDDING_PATH_new = get_detailed_path(self.ITEM_VAL_EMBEDDING_PATH, i)
            USER_VAL_EMBEDDING_PATH_new = get_detailed_path(self.USER_VAL_EMBEDDING_PATH, i)

            representation_save1 = save_embedding(model, df_item, item_columns, ITEM_EMBEDDING_PATH_new)
            representation_save2 = save_embedding(model, df_user, user_columns, USER_EMBEDDING_PATH_new)
            representation_save3 = save_embedding(model, df_item_val, item_columns, ITEM_VAL_EMBEDDING_PATH_new)
            representation_save4 = save_embedding(model, df_user_val, user_columns, USER_VAL_EMBEDDING_PATH_new)

        logzero.logger.info(f"user_model and its parameters have been saved in {self.MODEL_SAVE_PATH}")


def get_detailed_path(Path_old, num):
    path_list = Path_old.split(".")
    assert len(path_list) >= 2
    filename = path_list[-2]

    path_list_new = path_list[:-2] + [filename + f"_M{num}"] + path_list[-1:]
    Path_new = ".".join(path_list_new)
    return Path_new


def construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features):
    user_ids = np.unique(dataset_val.x_numpy[:, dataset_val.user_col].astype(int))
    item_ids = np.unique(dataset_val.x_numpy[:, dataset_val.item_col].astype(int))

    df_user_complete = pd.DataFrame(
        df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
        columns=df_user.reset_index()[user_features].columns)
    df_item_complete = pd.DataFrame(np.tile(df_item.loc[item_ids].reset_index()[item_features], (len(user_ids), 1)),
                                    columns=df_item.loc[item_ids].reset_index()[item_features].columns)

    df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)
    return df_x_complete


def get_one_predicted_res(model, df_x_complete, test_loader, steps_per_epoch):
    mean_all = []
    var_all = []
    with torch.no_grad():
        for _, (x, y) in tqdm(enumerate(test_loader), total=steps_per_epoch, desc="Predicting data..."):
            x = x.to(model.device).float()
            mean, log_var = model.forward(x)
            y_pred = mean.cpu().data.numpy()  # .squeeze()
            log_var_all_cat = log_var.cpu().data.numpy()
            var_all_cat = np.exp(log_var_all_cat)
            mean_all.append(y_pred)
            var_all.append(var_all_cat)
    mean_all_cat = np.concatenate(mean_all).astype("float64").reshape([-1])
    var_all_cat = np.concatenate(var_all).astype("float64").reshape([-1])

    # user_ids = np.sort(df_x_complete["user_id"].unique())

    num_user = len(df_x_complete["user_id"].unique())
    num_item = len(df_x_complete["item_id"].unique())

    if num_user != df_x_complete["user_id"].max() + 1:
        assert num_item != df_x_complete["item_id"].max() + 1
        lbe_user = LabelEncoder()
        lbe_item = LabelEncoder()

        lbe_user.fit(df_x_complete["user_id"])
        lbe_item.fit(df_x_complete["item_id"])

        mean_mat = csr_matrix(
            (
            mean_all_cat, (lbe_user.transform(df_x_complete["user_id"]), lbe_item.transform(df_x_complete["item_id"]))),
            shape=(num_user, num_item)).toarray()

        var_mat = csr_matrix(
            (var_all_cat, (lbe_user.transform(df_x_complete["user_id"]), lbe_item.transform(df_x_complete["item_id"]))),
            shape=(num_user, num_item)).toarray()
    else:
        assert num_item == df_x_complete["item_id"].max() + 1
        mean_mat = csr_matrix(
            (mean_all_cat, (df_x_complete["user_id"], df_x_complete["item_id"])),
            shape=(num_user, num_item)).toarray()

        var_mat = csr_matrix(
            (var_all_cat, (df_x_complete["user_id"], df_x_complete["item_id"])),
            shape=(num_user, num_item)).toarray()

    # minn = mean_mat.min()
    # maxx = mean_mat.max()
    # normed_mat = (mean_mat - minn) / (maxx - minn)
    # return normed_mat

    return mean_mat, var_mat
