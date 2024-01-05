# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 9:50 上午
# @Author  : Chongming GAO
# @FileName: user_model.py
import pickle
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.util.inputs import SparseFeatP, create_embedding_matrix
from src.core.util.layers import Linear


try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from deepctr_torch.inputs import build_input_features, DenseFeat, VarLenSparseFeat
from deepctr_torch.callbacks import History


class UserModel(nn.Module):
    def __init__(self, feature_columns, y_columns,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5,
                 l2_reg_dnn=0, init_std=0.0001, task_dnn_units=None, seed=2022, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu'):

        super(UserModel, self).__init__()

        torch.manual_seed(seed)

        self.feature_index = build_input_features(feature_columns)
        self.y_index = build_input_features(y_columns)

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device

        linear_feature_columns = feature_columns
        dnn_feature_columns = feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

        self.RL_eval_fun = None

        self.softmax = nn.Softmax(dim=0)

    def compile_RL_test(self, RL_eval_fun):
        self.RL_eval_fun = RL_eval_fun

    def compile(self, optimizer, loss_dict=None, metrics=None, metric_fun=None, loss_func=None, metric_fun_ranking=None):
        # metric_fun is a function!

        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.metrics = self._get_metrics(metrics)

        self.metric_fun = metric_fun
        self.metric_fun_ranking = metric_fun_ranking

        self.loss_dict = None if loss_dict is None else {x: self._get_loss_func(loss) if isinstance(loss, str) else loss
                                                         for x, loss in loss_dict.items()}  # deprecated!
        self.loss_func = loss_func

    def fit_data(self, dataset_train, dataset_val=None, batch_size=256, epochs=1, verbose=1, initial_epoch=0,
                 callbacks=None, shuffle=True):

        model = self.train()
        # loss_func_dict = self.loss_func_dict
        optim = self.optim

        # if self.gpus:
        #     print('parallel running on these gpus:', self.gpus)
        #     model = torch.nn.DataParallel(model, device_ids=self.gpus)
        #     batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        # else:
        #     print(self.device)
        print("Training device is [{}]".format(self.device))

        train_loader = DataLoader(
            dataset=dataset_train.get_dataset_train(), shuffle=shuffle, batch_size=batch_size,
            num_workers=dataset_train.num_workers)

        # train_loader = DataLoader(
        #     dataset=dataset_train.get_dataset_train(), shuffle=False, batch_size=batch_size)

        sample_num = len(dataset_train)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        # callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False


        epoch_logs = {}
        if dataset_val:
            eval_result = self.evaluate_data(dataset_val, batch_size)
            for name, result in eval_result.items():
                epoch_logs["val_" + name] = result
        if self.RL_eval_fun:
            eval_result_RL = self.RL_eval_fun(self.eval())
            for name, result in eval_result_RL.items():
                epoch_logs["RL_val_" + name] = result
        callbacks.on_epoch_end(-1, epoch_logs)

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(dataset_train), 0 if dataset_val is None else len(dataset_val), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            print("Training the {}/{} epoch".format(epoch, epochs))
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), total=steps_per_epoch) as t:
                    # for _, (x_user, x_item, neg_items, y_train) in t:
                    for i, (x, y, score) in t:
                        x = x.to(self.device).float()
                        y = y.to(self.device).float()
                        score = score.to(self.device).float()

                        loss = model.get_loss(x, y, score).squeeze()

                        optim.zero_grad()
                        # loss = loss_func(y_pred, y.squeeze(), reduction='sum')

                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        t.set_postfix(loss=f"{float(loss):.3f}", reg_loss=f"{float(reg_loss):.3f}")

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if bool(torch.isnan(total_loss)):
                            # print(i, x, y, score, loss, reg_loss, self.aux_loss)
                            model_parameters = {
                                "i": i,
                                "x": x,
                                "y": y,
                                "score": score,
                                "loss": loss,
                                "reg_loss": reg_loss,
                                "aux_loss": self.aux_loss
                            }
                            MODEL_SAVE_PATH = "debug_error.pkl"
                            with open(MODEL_SAVE_PATH, "wb") as output_file:
                                pickle.dump(model_parameters, output_file)
                            raise ("there is nan, please check {}".format(MODEL_SAVE_PATH))

                        # if verbose > 0:
                        #     for name, metric_fun in self.metrics.items():
                        #         if name not in train_result:
                        #             train_result[name] = []
                        #         train_result[name].append(metric_fun(
                        #             y.cpu().data.numpy(),
                        #             y_pred.cpu().data.numpy().astype("float64"),
                        #             x_user[:, 0].cpu().data.numpy().astype(int)
                        #         ))


            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num

            # for name, result in train_result.items():
            #     epoch_logs[name] = np.sum(result) / steps_per_epoch

            if dataset_val:
                eval_result = self.evaluate_data(dataset_val, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            if self.RL_eval_fun:
                eval_result_RL = self.RL_eval_fun(self.eval())
                for name, result in eval_result_RL.items():
                    epoch_logs["RL_val_" + name] = result

            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                # for name in self.metrics:
                #     eval_str += " - " + name + \
                #                 ": {0: .4f}".format(epoch_logs[name])

                if dataset_val:
                    for name in self.metric_fun:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def compile_UCB(self, n_arm):
        self.n_rec = n_arm
        self.n_each = np.ones(n_arm)

    def recommend_k_item(self, user, dataset_val, k=1, is_softmax=True, epsilon=0, is_ucb=False):  # for kuaishou data

        df_user_val = dataset_val.df_user_val
        df_item_val = dataset_val.df_item_val

        item_index = df_item_val.index.to_numpy()

        u_all_item = torch.tensor(
            np.concatenate((np.ones([len(df_item_val), 1]) * user,
                            df_user_val.loc[user].to_numpy() * np.array([[1]] * len(df_item_val)),
                            np.expand_dims(item_index, axis=-1),
                            df_item_val.values), 1),
            dtype=torch.float, device=self.device, requires_grad=False)

        assert u_all_item.shape[1] == len(dataset_val.x_columns)

        # # 用户的所有评分
        # if df_user is None:
        #     u_all_item = torch.tensor(
        #         np.concatenate((np.ones([len(df_item_val), 1]) * user,
        #                         np.expand_dims(item_index, axis=-1),
        #                         df_item_val.values), 1),
        #         dtype=torch.float, device=self.device, requires_grad=False)
        # else:
        #     u_all_item = torch.tensor(
        #         np.concatenate((np.ones([len(df_item_val), 1]) * user,
        #                         df_user.loc[user].to_numpy() * np.array([[1]] * len(df_item_val)),
        #                         np.expand_dims(item_index, axis=-1),
        #                         df_item_val.values), 1),
        #         dtype=torch.float, device=self.device, requires_grad=False)

        u_value = self.forward(u_all_item).detach().squeeze() # predicted value

        if is_ucb:
            if not hasattr(self, "n_rec"):
                self.compile_UCB(len(u_value))

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            ucb_bound = (2 * np.log(self.n_rec) / self.n_each) ** 0.5

            # ucb_bound[np.isnan(ucb_bound)] = 0
            # ucb_bound[np.isinf(ucb_bound)] = u_value.max()

            u_value_ucb = u_value + torch.Tensor(ucb_bound).to(u_value.device)
        else:
            u_value_ucb = u_value


        if is_softmax:
            value = self.softmax(u_value_ucb)
            index = torch.multinomial(value, k, replacement=False)
        else:
            # value = u_value
            # if min(value) < 0:
            #     value = -min(value) + value
            #     value = value / sum(value)
            # Todo:
            # index = u_value_ucb.argmax() # 预测分数的max

            # value = u_value_ucb/sum(u_value_ucb)
            # index = torch.multinomial(value, k, replacement=False)

            _, index = torch.topk(u_value_ucb, k)

        if epsilon > 0 and np.random.random() < epsilon:
            # # epsilon-greedy activated!!
            index = torch.randint(0, len(item_index), (k,))

        if is_ucb:
            self.n_rec += k
            self.n_each[index] += 1

        # print(int(index), end=' ')

        recommendation = item_index[index]
        value_rec = u_value.cpu().numpy()[index]

        return recommendation, value_rec

    def evaluate_data(self, dataset_val, batch_size=256):

        y_predict = self.predict_data(dataset_val, batch_size*10)
        y = dataset_val.get_y()

        eval_result = {}
        for name, metric_fun in self.metric_fun.items():
            eval_result[name] = metric_fun(y, y_predict)

        if self.metric_fun_ranking is not None:
            ground_truth = dataset_val.ground_truth

            if dataset_val.all_item_ranking:
                y_complete_predict = self.predict_data(dataset_val.dataset_complete, batch_size*10)

                user_id = dataset_val.dataset_complete.x_numpy[:,dataset_val.user_col]
                item_id = dataset_val.dataset_complete.x_numpy[:,dataset_val.item_col]

                # xy_predict = pd.DataFrame([user_id, item_id, y_predict.squeeze()], columns={"user_id", "item_id", "y_pred"})
                xy_predict = pd.DataFrame({"user_id":user_id, "item_id":item_id, "y_pred":y_complete_predict.squeeze()})
            else:
                user_id = dataset_val.x_numpy[:, dataset_val.user_col]
                item_id = dataset_val.x_numpy[:, dataset_val.item_col]
                xy_predict = pd.DataFrame(
                    {"user_id": user_id, "item_id": item_id, "y_pred": y_predict.squeeze()})

            xy_predict["y_true"] = y
            xy_predict = xy_predict.astype(dtype={"user_id": "int64", "item_id": "int64", "y_pred": "float64"})
            # df_score = xy_predict.groupby("user_id").agg(list)

            eval_result.update(self.metric_fun_ranking(xy_predict, ground_truth))

        return eval_result

    def predict_data(self, dataset_predict, batch_size=256, verbose=False):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()

        # dataset_predict.set_only_x(True)

        # test_loader = DataLoader(dataset=dataset_predict, shuffle=False, batch_size=batch_size)
        # test_loader = DataLoader(dataset=dataset_predict.get_dataset_eval(), shuffle=False, batch_size=batch_size)

        is_shuffle=False
        assert not is_shuffle
        test_loader = DataLoader(dataset=dataset_predict.get_dataset_eval(), shuffle=is_shuffle, batch_size=batch_size,
                                 num_workers=dataset_predict.num_workers)

        sample_num = len(dataset_predict)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        pred_ans = []
        with torch.no_grad():
            # for _, x_test in enumerate(test_loader):
            for _, (x, y) in tqdm(enumerate(test_loader), total=steps_per_epoch, desc="Predicting data..."):
                # if isinstance(x_test, list):
                #     for i in range(len(x_test)):
                #         if isinstance(x_test[i], list):
                #             for j in range(len(x_test[i])):
                #                 x_test[i][j] = x_test[i][j].to(self.device).float()
                #         elif isinstance(x_test[i], torch.Tensor):
                #             x_test[i] = x_test[i].to(self.device).float()
                # else:
                #     raise Exception("No!")

                x = x.to(self.device).float()
                # y = y.to(self.device).float()

                y_pred = model.forward(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeatP, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accurac y" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def save_model_embedding(self, ):
        self.embedding_dict

