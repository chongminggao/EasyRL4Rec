from abc import ABC
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tianshou.policy import BasePolicy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tianshou.data import Batch, ReplayBuffer

class RecPolicy(ABC, nn.Module):
    @staticmethod
    def parse_model_args(parser):
        # parser.add_argument('--discrete_action_hidden', type=int, nargs='+', default=[128], 
        #                     help='hidden dim of the action net')
        parser.add_argument('--slate_size', type=int, default=1,
                            help='slate size for actions')
        
        return parser
    
    def __init__(
        self,
        args,
        policy: BasePolicy,
        state_tracker
    ) -> None:
        super().__init__()
        self.policy = policy
        self.action_type = policy.action_type

        self.state_tracker = state_tracker
        self.n_items = state_tracker.num_item
        self.emb_dim = state_tracker.emb_dim
        self.device = args.device

        # self.slate_size = args.slate_size
        self.slate_size = 1

    def get_score(self, action_emb, do_softmax = False):
        '''
        score = dot(action_emb, item_embs)
        
        @input:
        - action_emb: [B, emb_dim]
        '''
        item_index = np.expand_dims(np.arange(self.n_items), -1)  # [n_items, 1]
        item_embs = self.state_tracker.get_embedding(item_index, "action")  # [n_items, emb_dim]

        item_embs = F.normalize(item_embs, dim=-1)  # 归一化
        batch_item_embs = item_embs.repeat(action_emb.shape[0],1,1)  # [B, n_items, emb_dim]
        
        action_emb = torch.Tensor(action_emb).to(self.device)
        scores = torch.sum(action_emb.view(-1,1,self.emb_dim) * batch_item_embs, dim = -1)  # [B, n_items]

        if do_softmax:
            return torch.softmax(scores, dim = -1)
        else:
            return scores
        
    def select_action(self, action_scores, deterministic=True):
        # two types of greedy selection
        if not deterministic and np.random.rand() >= self.topk_rate:
            # e-greedy dist sample
            # TODO
            pass
        else:
            _, indices = torch.topk(action_scores, k = self.slate_size, dim = 1)
            item_ids = torch.arange(self.n_items).to(self.device)
            action = item_ids[indices].detach()  # [B, slate_size]

            action = action.squeeze(1)  # item-wise
        return np.array(action.cpu())
    
    def forward(
        self,
        batch: Batch,
        buffer: Optional[ReplayBuffer],
        indices: np.ndarray = None,
        is_obs = None,
        remove_recommended_ids = False,
        is_train = True, 
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        use_batch_in_statetracker = False,
        **kwargs: Any,
    ) -> Batch:
        # batch.obs = self.state_tracker(buffer=buffer, indices=indices, is_seq=is_seq, batch=batch, is_train=is_train, use_batch_in_statetracker=use_batch_in_statetracker)
        batch.mask = self._get_recommend_mask(remove_recommended_ids, batch.obs.shape[0], buffer)
        batch = self.policy(batch=batch, 
                            buffer=buffer,
                            indices=indices, 
                            is_obs=is_obs,
                            remove_recommended_ids=remove_recommended_ids,
                            is_train=is_train,
                            state=state,
                            input=input,
                            use_batch_in_statetracker=use_batch_in_statetracker,
                            **kwargs)
        
        return batch
    
    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)  # RecPolicy.process_fn()
        result = self.policy.learn(batch, **kwargs)

        self.policy.post_process_fn(batch, buffer, indices)
        if self.policy.lr_scheduler is not None:
            self.policy.lr_scheduler.step()
        self.policy.updating = False
        return result

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        self.policy._buffer, self.policy._indices = buffer, indices
        # calculate batch.obs&obs_next using state_tracker for policy to 'learn'
        # batch.obs = self.state_tracker(buffer, indices, is_seq=True)
        # batch.obs_next = self.state_tracker(buffer, indices, is_seq=False)
        batch.mask = self._get_recommend_mask(False, batch.obs.shape[0], buffer)
        batch = self.policy.process_fn(batch, buffer, indices)
        return batch

    # collector related
    def map_action_inverse(
        self, act: Union[Batch, List, np.ndarray]
    ) -> Union[Batch, List, np.ndarray]:
        return self.policy.map_action_inverse(act)

    def map_action(
            self, 
            batch: Batch
        )-> Union[Batch, np.ndarray]:
        act = self.policy.map_action(batch.act)
        if self.action_type == "continuous":
            action_scores = self.get_score(act)  # [B, n_items]
            # TODO: remove recommended item id

            discrete_acts = self.select_action(action_scores)
        else:
            discrete_acts = act
        return discrete_acts

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        return self.policy.exploration_noise(act, batch)
    
    # Trainer related
    def state_dict(self):
        return self.policy.state_dict()

    def train(self, mode: bool = True):
        self.policy.train()

    def eval(self, mode: bool = True):
        self.policy.eval()

    def _get_recommend_mask(self, remove_recommended_ids, batch_size, buffer):
        mask = torch.ones((batch_size, self.n_items), dtype=torch.bool).to(device=self.device)
        recommended_ids = get_recommended_ids(buffer) if remove_recommended_ids else None
        if recommended_ids is not None:
            recommended_ids_torch = torch.LongTensor(recommended_ids).to(device=self.device)
            mask = mask.scatter(1, recommended_ids_torch, 0)
        return mask.detach()

def get_recommended_ids(buffer):
    if len(buffer) == 0:
        recommended_ids = None
    else:
        indices = buffer.last_index[~buffer[buffer.last_index].done]

        # is_alive = True
        recommended_ids = np.zeros([0, len(indices)], dtype=int)
        while True:
            acts = buffer.obs_next[indices][:, 1]  # buffer.act[indices]
            recommended_ids = np.vstack([recommended_ids, acts])

            if all(indices == buffer.prev(indices)):
                break
            assert all(indices != buffer.prev(indices))

            indices = buffer.prev(indices)

        recommended_ids = recommended_ids.T
    return recommended_ids