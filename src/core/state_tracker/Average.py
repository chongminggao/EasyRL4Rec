from src.core.state_tracker.base import StateTracker_Base
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.core.util.utils import compute_input_dim

FLOAT = torch.FloatTensor


class StateTrackerAvg(StateTracker_Base):
    def __init__(self, 
                 user_columns, action_columns, feedback_columns, 
                 dim_model,
                 train_max=None, 
                 train_min=None, 
                 test_max=None, 
                 test_min=None, 
                 reward_handle=None,
                 saved_embedding=None,
                 device="cpu", 
                 use_userEmbedding=False, window_size=10):
        
        assert saved_embedding is not None
        super(StateTrackerAvg, self).__init__(
            user_columns=user_columns, 
            action_columns=action_columns,
            feedback_columns=feedback_columns,
            dim_model=dim_model,
            train_max=train_max,
            train_min=train_min,
            test_max=test_max,
            test_min=test_min,
            reward_handle=reward_handle,
            saved_embedding=saved_embedding,
            device=device,
            window_size=window_size, 
            use_userEmbedding=use_userEmbedding)

        self.final_dim = self.hidden_size

        

    def forward(self, buffer=None, indices=None, is_obs=None, batch=None, is_train=True, use_batch_in_statetracker=False, **kwargs):

        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices=indices, is_obs=is_obs, batch=batch, use_batch_in_statetracker=use_batch_in_statetracker, is_train=is_train)

        state_sum = seq.sum(dim=1)
        state_final = state_sum / torch.from_numpy(np.expand_dims(len_states, -1)).to(self.device)
        
        return state_final