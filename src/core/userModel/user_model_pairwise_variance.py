# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 3:31 下午
# @Author  : Chongming GAO
# @FileName: user_model_pairwise.py

import torch
from deepctr_torch.inputs import combined_dnn_input, build_input_features
from deepctr_torch.layers import DNN, PredictionLayer, FM
from torch import nn
import inspect

from core.util.inputs import input_from_feature_columns
from core.util.layers import Linear, create_embedding_matrix
from core.userModel.user_model_variance import UserModel_Variance
from core.userModel.utils import compute_input_dim

class UserModel_Pairwise_Variance(UserModel_Variance):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"``

    :return: A PyTorch model instance.
    """

    def __init__(self, feature_columns, y_columns, task, task_logit_dim,
                 dnn_hidden_units=(128, 128), dnn_hidden_units_var=(),
                 l2_reg_embedding=1e-5, l2_reg_dnn=1e-1, init_std=0.0001, task_dnn_units=None, seed=2022, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', ab_columns=None,
                 max_logvar=0.5, min_logvar=-10):


        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)

        self.model_param = {key: values[key] for key in args[1:]}

        super(UserModel_Pairwise_Variance, self).__init__(feature_columns, y_columns,
                                                          l2_reg_embedding=l2_reg_embedding,
                                                          init_std=init_std, seed=seed, device=device)

        self.max_logvar = max_logvar
        self.min_logvar = min_logvar

        self.feature_columns = feature_columns
        self.feature_index = self.feature_index

        self.y_columns = y_columns
        self.task_logit_dim = task_logit_dim

        self.sigmoid = nn.Sigmoid()
        """
        For MMOE Layer
        """
        self.task = task
        self.task_dnn_units = task_dnn_units

        """
        For DNN Layer.
        """

        self.dnn = DNN(compute_input_dim(self.feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.last = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        # self.out = PredictionLayer(task, task_dim=1)
        self.out = PredictionLayer(task)

        if len(dnn_hidden_units_var) > 0:
            self.layers_var = DNN(dnn_hidden_units[-1], dnn_hidden_units_var,
                                  activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                  use_bn=dnn_use_bn,
                                  init_std=init_std, device=device)

            self.last_var = nn.Linear(dnn_hidden_units_var[-1], 1, bias=False)
        else:
            self.layers_var = None
            self.last_var = nn.Linear(dnn_hidden_units[-1], 1, bias=False)

        """
        For FM Layer.
        """
        use_fm = True if task_logit_dim == 1 else False
        self.use_fm = use_fm

        self.fm_task = FM() if use_fm else None

        self.linear = Linear(self.feature_columns, self.feature_index, device=device)

        """
        For exposure effect
        """
        if ab_columns is not None:
            ab_embedding_dict = create_embedding_matrix(ab_columns, init_std, sparse=False, device=device)
            for tensor in ab_embedding_dict.values():
                nn.init.normal_(tensor.weight, mean=1, std=init_std)

            self.ab_embedding_dict = ab_embedding_dict

        self.ab_columns = ab_columns

        self.add_regularization_weight(self.parameters(), l2=l2_reg_dnn)

        self.to(device)

    def _deepfm(self, X, feature_columns, feature_index):

        # sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, feature_columns,
        #                                                                           self.embedding_dict,
        #                                                                           feature_index=feature_index)
        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, feature_columns, self.embedding_dict,
                                                                             feature_index,
                                                                             support_dense=True, device=self.device)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        linear_model = self.linear
        dnn = self.dnn
        last = self.last
        out = self.out

        # Linear and FM logit
        logit = torch.zeros([len(X), self.task_logit_dim], device=X.device)

        if linear_model is not None:
            logit = logit + linear_model(X)

            fm_model = self.fm_task
            if self.use_fm and len(sparse_embedding_list) > 0 and fm_model is not None:
                fm_input = torch.cat(sparse_embedding_list, dim=1)
                logit += fm_model(fm_input)

        linear_logit = logit

        # DNN
        dnn_output = dnn(dnn_input)
        dnn_logit = last(dnn_output)
        all_logit = linear_logit + dnn_logit
        y_pred = out(all_logit)

        if self.layers_var is not None:
            var_output = self.layers_var(dnn_output)
        else:
            var_output = dnn_output
        log_var = self.last_var(var_output)

        # todo: 放缩 【-10, 0.5]
        softplus = nn.Softplus()
        log_var = self.max_logvar - softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + softplus(log_var - self.min_logvar)

        return y_pred, log_var

    def get_loss(self, x, y, score, deterministic=False):
        # Split positive and negative samples
        assert x.shape[1] % 2 == 0
        num_features = x.shape[1] // 2
        X_pos = x[:, :num_features]
        X_neg = x[:, num_features:]

        # y_deepfm_pos = self._deepfm(X_pos, self.feature_columns, self.feature_index)
        # y_deepfm_neg = self._deepfm(X_neg, self.feature_columns, self.feature_index)
        y_deepfm_pos, log_var_pos = self.forward(X_pos)
        y_deepfm_neg, log_var_neg = self.forward(X_neg)

        if self.ab_columns is None:
            alpha_u, beta_i = None, None
        else:  # CIRS-UserModel-kuaishou.py
            alpha_u = self.ab_embedding_dict['alpha_u'](x[:, 0].long())
            beta_i = self.ab_embedding_dict['beta_i'](x[:, 1].long())

        if not deterministic:
            loss = self.loss_func(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=alpha_u, beta_i=beta_i,
                                  log_var=log_var_pos, log_var_neg=log_var_neg)
        else:
            loss = self.loss_func(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=alpha_u, beta_i=beta_i, log_var=None)

        return loss

    def forward(self, x):
        y_deepfm, var = self._deepfm(x, self.feature_columns, self.feature_index)
        return y_deepfm, var
