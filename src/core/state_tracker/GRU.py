from src.core.state_tracker.base import StateTracker_Base

import torch
from torch import nn

FLOAT = torch.FloatTensor


class StateTracker_GRU(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, 
                 dim_model,
                 train_max=None, 
                 train_min=None, 
                 test_max=None, 
                 test_min=None, 
                 reward_handle=None,
                 saved_embedding=None,
                 device="cpu",
                 use_userEmbedding=False, window_size=10, gru_layers=1):
        super().__init__(user_columns=user_columns, action_columns=action_columns,
                         feedback_columns=feedback_columns,
                         dim_model=dim_model,
                         train_max=train_max,
                         train_min=train_min,
                         test_max=test_max,
                         test_min=test_min,
                         reward_handle=reward_handle,
                         saved_embedding=saved_embedding,
                         device=device,
                         window_size=window_size, use_userEmbedding=use_userEmbedding)

        self.final_dim = self.hidden_size

        # Horizontal Convolutional Layers
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )

    def forward(self, buffer=None, indices=None, is_obs=None, batch=None, is_train=True, use_batch_in_statetracker=False, **kwargs):

        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices=indices, is_obs=is_obs, batch=batch, use_batch_in_statetracker=use_batch_in_statetracker, is_train=is_train)

        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(seq, len_states, batch_first=True, enforce_sorted=False)

        emb_packed_final, hidden = self.gru(emb_packed)
        hidden_final = hidden.view(-1, hidden.shape[2])

        return hidden_final