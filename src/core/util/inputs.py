# -*- coding: utf-8 -*-
# @Time    : 2021/10/6 9:58 下午
# @Author  : Chongming GAO
# @FileName: inputs.py

# from collections import namedtuple
from torch import nn

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, varlen_embedding_lookup, \
    get_varlen_pooling_list

DEFAULT_GROUP_NAME = "default_group"


class SparseFeatP(SparseFeat):
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, padding_idx=None):
        return super(SparseFeatP, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                               embedding_name, group_name)

    def __init__(self, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, padding_idx=None):
        self.padding_idx = padding_idx



def get_dataset_columns(dim_user, dim_action, num_user, num_action, envname="VirtualTB-v0"):
    user_columns, action_columns, feedback_columns = [], [], []
    has_user_embedding, has_action_embedding, has_feedback_embedding = None, None, None
    if envname == "VirtualTB-v0":
        user_columns = [DenseFeat("feat_user", 88)]
        action_columns = [DenseFeat("feat_item", 27)]
        # feedback_columns = [SparseFeat("feat_feedback", 11, embedding_dim=27)]
        feedback_columns = [DenseFeat("feat_feedback", 1)]
        has_user_embedding = True
        has_action_embedding = True
        has_feedback_embedding = True
    else: # for kuairecenv, coat, yahoo
        user_columns = [SparseFeatP("feat_user", num_user, embedding_dim=dim_user)]
        action_columns = [SparseFeatP("feat_item", num_action, embedding_dim=dim_action)]
        feedback_columns = [DenseFeat("feat_feedback", 1)]
        has_user_embedding = False
        has_action_embedding = False
        has_feedback_embedding = True

    return user_columns, action_columns, feedback_columns, \
           has_user_embedding, has_action_embedding, has_feedback_embedding


def input_from_feature_columns(X, feature_columns, embedding_dict, feature_index, support_dense: bool, device):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError(
            "DenseFeat is not supported in dnn_feature_columns")

    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
        feat in sparse_feature_columns]

    sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                  varlen_sparse_feature_columns)
    varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                           varlen_sparse_feature_columns, device)

    dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                        dense_feature_columns]

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse,
                                           padding_idx=feat.padding_idx)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        if tensor.padding_idx is None:
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        else:
            nn.init.normal_(tensor.weight[:tensor.padding_idx], mean=0, std=init_std)
            nn.init.normal_(tensor.weight[tensor.padding_idx+1:], mean=0, std=init_std)

    return embedding_dict.to(device)


