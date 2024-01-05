# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 3:31 下午
# @Author  : Chongming GAO
# @FileName: user_model_mmoe.py

import torch

from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, PredictionLayer, FM
from torch import nn

from src.core.util.inputs import input_from_feature_columns, create_embedding_matrix
from src.core.util.layers import MMOELayer, Linear
from src.core.userModel.user_model import UserModel
from src.core.util.utils import compute_input_dim

class UserModel_MMOE(UserModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
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

    def __init__(self, feature_columns, y_columns,
                 num_tasks, tasks, task_logit_dim,
                 num_experts=4,
                 expert_dim=8,
                 dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=1e-2, init_std=0.0001, task_dnn_units=None, seed=2021, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', padding_idx=None, ab_columns=None):

        # self.is_ssl = is_ssl
        # feature_columns_dict = {i.name: i for i in ssl_columns}
        # task_columns = [feature_columns_dict[name] for name in task_features]

        super(UserModel_MMOE, self).__init__(feature_columns, y_columns,
                                             l2_reg_embedding=l2_reg_embedding,
                                             init_std=init_std, seed=seed, device=device,
                                             padding_idx=padding_idx)

        # task_indeces = [ssl_columns.index(feat) for feat in self.feature_index]
        # task_indeces = list(itertools.chain(*[(range(*self.feature_index[feat])) for feat in task_features]))

        # self.task_feature_index = build_input_features(task_columns)

        # self.ssl_columns = ssl_columns
        # self.task_columns = task_columns

        # self.task_indeces = task_indeces
        # self.user_features = user_features
        # self.item_features = item_features

        # assert [i.name for i in ssl_columns] == user_features + item_features

        self.feature_columns = feature_columns
        self.y_columns = y_columns
        self.task_logit_dim = task_logit_dim

        """
        For MMOE Layer
        """
        self.tasks = tasks
        self.task_dnn_units = task_dnn_units

        self.dnn = DNN(compute_input_dim(feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.mmoe_layer = MMOELayer(dnn_hidden_units[-1], num_tasks, num_experts, expert_dim)

        if task_dnn_units is not None:
            # the last layer of task_dnn should be expert_dim
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units + (expert_dim,)) for _ in range(num_tasks)])
        # self.tower_network = nn.ModuleList([nn.Linear(expert_dim, 1, bias=False) for _ in range(num_tasks)])
        self.tower_network = nn.ModuleList(
            [nn.Linear(expert_dim, task_dim, bias=False) for name, task_dim in task_logit_dim.items()])
        self.out = nn.ModuleList(
            [PredictionLayer(self.tasks[name], task_dim) for name, task_dim in self.task_logit_dim.items()])

        # """
        # For dnn Layer
        # """
        # self.dnn = DNN(compute_input_dim(feature_columns), dnn_hidden_units,
        #                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
        #                init_std=init_std, device=device)
        # self.dnn_linear = nn.Linear(
        #     dnn_hidden_units[-1], 1, bias=False).to(device)
        #
        # self.add_regularization_weight(
        #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        # self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        """
        For DeepFM Layer.
        """
        use_fm = {name: True if task_dim == 1 else False for name, task_dim in task_logit_dim.items()}
        self.use_fm = use_fm

        self.fm_task = nn.ModuleList([FM() if is_use else None for name, is_use in use_fm.items()])

        self.linear_model_task = nn.ModuleList(
            [Linear(feature_columns, self.feature_index, device=device) if task_dim == 1 else None for name, task_dim in
             task_logit_dim.items()])

        # if use_fm:
        #     self.fm_task = nn.ModuleList([FM() for _ in range(num_tasks)])

        # self.linear_model_task = nn.ModuleList(
        #     [Linear(feature_columns, self.feature_index, device=device) for _ in range(num_tasks)])

        # add regularization
        self.add_regularization_weight(self.parameters(), l2=l2_reg_dnn)

        """
        For exposure effect
        """
        if ab_columns is not None:
            ab_embedding_dict = create_embedding_matrix(ab_columns, init_std, sparse=False, device=device)
            for tensor in ab_embedding_dict.values():
                nn.init.normal_(tensor.weight, mean=1, std=init_std)

            self.ab_embedding_dict = ab_embedding_dict

        self.ab_columns = ab_columns

        self.to(device)

    def _mmoe(self, X, is_sigmoid):
        # sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X,
        #                                                                           self.feature_columns,
        #                                                                           self.embedding_dict)
        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, self.feature_columns,
                                                                             self.embedding_dict, self.feature_index,
                                                                             support_dense=True, device=self.device)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        mmoe_outs = self.mmoe_layer(dnn_output)
        if self.task_dnn_units is not None:
            mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]

        task_outputs = []
        for i, mmoe_out in enumerate(mmoe_outs):
            logit = self.tower_network[i](mmoe_out)

            if is_sigmoid:
                output = self.out[i](logit)
                task_outputs.append(output)
            else:
                task_outputs.append(logit)

        task_outputs = torch.cat(task_outputs, -1)
        return task_outputs

    def _deepfm(self, x):

        X = x
        # sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X,
        #                                                                           self.feature_columns,
        #                                                                           self.embedding_dict)
        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, self.feature_columns,
                                                                             self.embedding_dict, self.feature_index,
                                                                             support_dense=True, device=self.device)

        # Linear and FM logit
        linear_logit_list = []
        for i, name in enumerate(self.tasks):
            logit = torch.zeros([len(X), self.task_logit_dim[name]], device=X.device)

            linear_model = self.linear_model_task[i]
            if linear_model is not None:
                logit = logit + linear_model(X)

            fm_model = self.fm_task[i]
            if self.use_fm and len(sparse_embedding_list) > 0 and fm_model is not None:
                fm_input = torch.cat(sparse_embedding_list, dim=1)
                logit += fm_model(fm_input)

            linear_logit_list.append(logit)

        linear_logit = torch.cat(linear_logit_list, -1)

        # MMOE
        dnn_logit = self._mmoe(X, is_sigmoid=False)  # For more than one task

        # # Dnn
        # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        # dnn_output = self.dnn(dnn_input)
        # dnn_logit = self.dnn_linear(dnn_output)

        logit = linear_logit + dnn_logit

        output_list = []
        for i, (name, taskloss) in enumerate(self.tasks.items()):
            logit_i = logit[:, self.y_index[name][0]:self.y_index[name][1]]
            output = self.out[i](logit_i)
            # output = output.unsqueeze(-1)
            output_list.append(output)

            # output = self.out[i](logit[:, i])
            # output = output.unsqueeze(-1)
            # output_list.append(output)

        y_pred = torch.cat(output_list, -1)

        # y_score = y_pred * score if score is not None else y_pred
        y_score = y_pred

        return y_score

    def get_loss(self, x, y, score):

        y_deepfm = self._deepfm(x)

        if self.ab_columns is None:
            loss = self.loss_func(y_deepfm, y, score, self.y_index)
        else:  # CIRS-UserModel-kuaishou.py
            alpha_u = self.ab_embedding_dict['alpha_u'](x[:, 0].long())
            beta_i = self.ab_embedding_dict['beta_i'](x[:, 1].long())
            loss = self.loss_func(y_deepfm, y, score, alpha_u=alpha_u, beta_i=beta_i)

        return loss

    # def ssl_predict(self, x_user, x_item):
    #     pass

    # def forward(self, x_user, x_item, neg_items=None, y=1):
    #     if neg_items is not None:
    #         loss = self.ssl_train(x_user, x_item, neg_items, y)
    #         return loss
    #     else: # predict!
    #         y_deepfm = self._deepfm(x_user, x_item)
    #         return y_deepfm

    def forward(self, x):
        y_deepfm = self._deepfm(x)
        return y_deepfm
