from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from core.policy.utils import get_emb, get_recommended_ids, removed_recommended_id_from_embedding
from tianshou.data import Batch, to_torch, to_numpy, ReplayBuffer
from tianshou.policy import DiscreteCQLPolicy


class DiscreteCQLPolicy_withEmbedding(DiscreteCQLPolicy):
    """Implementation of discrete Conservative Q-Learning algorithm. arXiv:2006.04779.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int num_quantiles: the number of quantile midpoints in the inverse
        cumulative distribution function of the value. Default to 200.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param float min_q_weight: the weight for the cql loss.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::
        Please refer to :class:`~tianshou.policy.QRDQNPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        num_quantiles: int = 200,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        min_q_weight: float = 10.0,
        state_tracker=None,
        buffer=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model, optim, discount_factor, num_quantiles, estimation_step, target_update_freq, reward_normalization,
            min_q_weight, **kwargs
        )
        self.state_tracker = state_tracker
        self.buffer = buffer

    def forward(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray = None,
        is_obs=None,
        remove_recommended_ids=False,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        use_batch_in_statetracker = False,
        **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)

        obs_emb = get_emb(self.state_tracker, buffer, indices=indices, batch=batch, is_obs=is_obs, use_batch_in_statetracker=use_batch_in_statetracker)
        recommended_ids = get_recommended_ids(buffer) if remove_recommended_ids else None

        logits, hidden = model(obs_emb, state=state, info=batch.info)

        assert not hasattr(batch.obs, "obs")
        # obs = batch[input]
        # obs_next = obs.obs if hasattr(obs, "obs") else obs
        # logits, hidden = model(obs_next, state=state, info=batch.info)


        q = self.compute_q_value(logits, getattr(obs_emb, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]

        logits_masked, indices_masked = removed_recommended_id_from_embedding(q, recommended_ids)
        # act = to_numpy(q.max(dim=1)[1])
        act = logits_masked.max(dim=1)[1]

        act_unsqueezed = act.unsqueeze(-1)
        act_ori = indices_masked.gather(dim=1, index=act_unsqueezed).squeeze(1)

        return Batch(logits=logits, act=act_ori, state=hidden)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        if self._target:
            act = self(batch, buffer, indices, is_obs=False, input="obs_next").act
            next_dist = self(batch, buffer, indices, is_obs=False, model="model_old", input="obs_next").logits
        else:
            next_batch = self(batch, buffer, indices, is_obs=False, input="obs_next")
            act = next_batch.act
            next_dist = next_batch.logits
        next_dist = next_dist[np.arange(len(act)), act, :]
        return next_dist  # shape: [bsz, num_quantiles]

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)



        all_dist = self(batch, self.buffer, indices=batch.indices, is_obs=True).logits
        # all_dist = self(batch).logits
        act = to_torch(batch.act, dtype=torch.long, device=all_dist.device)
        curr_dist = all_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            dist_diff *
            (self.tau_hat - (target_dist - curr_dist).detach().le(0.).float()).abs()
        ).sum(-1).mean(1)
        qr_loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = dist_diff.detach().abs().sum(-1).mean(1)  # prio-buffer
        # add CQL loss
        q = self.compute_q_value(all_dist, None)
        dataset_expec = q.gather(1, act.unsqueeze(1)).mean()
        negative_sampling = q.logsumexp(1).mean()
        min_q_loss = negative_sampling - dataset_expec
        loss = qr_loss + min_q_loss * self._min_q_weight
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {
            "loss": loss.item(),
            "loss/qr": qr_loss.item(),
            "loss/cql": min_q_loss.item(),
        }

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation.

        :param int sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.

        :return: A dict, including the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        batch.indices = indices
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return result