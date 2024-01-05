# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 11:52 上午
# @Author  : Chongming GAO
# @FileName: layers.py
import math

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch.nn.functional as F


from src.core.util.inputs import SparseFeatP
from src.core.util.inputs import create_embedding_matrix
from deepctr_torch.inputs import DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, varlen_embedding_lookup
from torch import Tensor


class Actor_Linear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_shape: int,
        device: Union[str, int, torch.device] = "cpu",
        softmax_output: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.output_dim = action_shape
        self.last = nn.Linear(input_dim, action_shape)
        self.softmax_output = softmax_output

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        logits = self.last(obs)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        hidden = None
        return logits, hidden


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit

class MMOELayer(nn.Module):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size,units)``.

      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, output_dim)`` .

      Arguments
        - **input_dim** : Positive integer, dimensionality of input features.
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **output_dim**: integer, the dimension of each output of MMOELayer.

    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, input_dim, num_tasks, num_experts, output_dim):
        super(MMOELayer, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.expert_network = nn.Linear(self.input_dim, self.num_experts * self.output_dim, bias=True)
        self.gating_networks = nn.ModuleList(
            [nn.Linear(self.input_dim, self.num_experts, bias=False) for _ in range(self.num_tasks)])
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)

    def forward(self, inputs):
        outputs = []
        expert_out = self.expert_network(inputs)
        expert_out = expert_out.reshape([-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = self.gating_networks[i](inputs)
            gate_out = gate_out.softmax(1).unsqueeze(-1)
            output = torch.bmm(expert_out, gate_out).squeeze()
            outputs.append(output)
        return outputs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
