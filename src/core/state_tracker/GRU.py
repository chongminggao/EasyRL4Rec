from core.state_tracker.base import StateTracker_Base

import torch
from torch import nn

FLOAT = torch.FloatTensor


class StateTracker_GRU(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device,
                 use_userEmbedding=False, window_size=10, gru_layers=1):
        super().__init__(user_columns=user_columns, action_columns=action_columns,
                         feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                         window_size=window_size)

        self.hidden_size = action_columns[0].embedding_dim
        self.num_item = action_columns[0].vocabulary_size

        self.final_dim = self.hidden_size

        # Item embedding
        embedding_dict = torch.nn.ModuleDict(
            {"feat_item": torch.nn.Embedding(num_embeddings=self.num_item + 1,
                                             embedding_dim=self.hidden_size)})
        self.embedding_dict = embedding_dict.to(device)
        nn.init.normal_(self.embedding_dict.feat_item.weight, mean=0, std=0.1)

        # Horizontal Convolutional Layers
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, **kwargs):

        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices, obs, reset, is_obs)

        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(seq, len_states, batch_first=True, enforce_sorted=False)

        emb_packed_final, hidden = self.gru(emb_packed)
        hidden_final = hidden.view(-1, hidden.shape[2])

        return hidden_final