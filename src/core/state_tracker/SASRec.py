from core.state_tracker.base import StateTracker_Base, extract_axis_1

import torch
from torch import nn
import torch.nn.functional as F

FLOAT = torch.FloatTensor


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]

        :return: A 3d tensor with shape of (N, T_q, C)

        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)

        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)

        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)

        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])  # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings,
                                       matmul_output_m1)  # (h*N, T_q, T_k)

        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)

        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask

        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)

        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)

        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual Connection
        output_res = output + queries

        return output_res


class StateTracker_SASRec(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device,
                 use_userEmbedding=False, window_size=10, dropout_rate=0.1, num_heads=1):
        super().__init__(user_columns=user_columns, action_columns=action_columns,
                         feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                         window_size=window_size)

        self.hidden_size = action_columns[0].embedding_dim
        self.num_item = action_columns[0].vocabulary_size
        self.dropout_rate = dropout_rate

        self.final_dim = self.hidden_size

        # Item embedding
        embedding_dict = torch.nn.ModuleDict(
            {"feat_item": torch.nn.Embedding(num_embeddings=self.num_item + 1,
                                             embedding_dim=self.hidden_size)})
        self.embedding_dict = embedding_dict.to(device)
        nn.init.normal_(self.embedding_dict.feat_item.weight, mean=0, std=0.1)

        self.positional_embeddings = nn.Embedding(
            num_embeddings=window_size,
            embedding_dim=self.hidden_size
        )
        nn.init.normal_(self.positional_embeddings.weight, 0, 0.01)

        # Supervised Head Layers
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        self.ln_2 = nn.LayerNorm(self.hidden_size)
        self.ln_3 = nn.LayerNorm(self.hidden_size)
        self.mh_attn = MultiHeadAttention(self.hidden_size, self.hidden_size, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(self.hidden_size, self.hidden_size, dropout_rate)


    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, **kwargs):

        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices, obs, reset, is_obs)
        inputs_emb = seq * self.hidden_size ** 0.5	
        inputs_pos_emb = inputs_emb + self.positional_embeddings(torch.arange(self.window_size).to(self.device))	
        seq = self.emb_dropout(inputs_pos_emb)

        seq *= mask
        # assert (seq == seq * mask).all()
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out_masked = ff_out * mask
        ff_out_3 = self.ln_3(ff_out_masked)

        # state_final = ff_out_3[:, 0, :]
        state_final = extract_axis_1(ff_out_3, len_states - 1).squeeze(1)

        return state_final
