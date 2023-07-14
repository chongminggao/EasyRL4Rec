from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from core.policy.utils import get_emb, get_recommended_ids, removed_recommended_id_from_embedding
from tianshou.data import Batch, to_torch, to_torch_as, ReplayBuffer
from tianshou.policy import DiscreteCRRPolicy


class DiscreteCRRPolicy_withEmbedding(DiscreteCRRPolicy):
    r"""Implementation of discrete Critic Regularized Regression. arXiv:2006.15134.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the action-value critic (i.e., Q function)
        network. (s -> Q(s, \*))
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param str policy_improvement_mode: type of the weight function f. Possible
        values: "binary"/"exp"/"all". Default to "exp".
    :param float ratio_upper_bound: when policy_improvement_mode is "exp", the value
        of the exp function is upper-bounded by this parameter. Default to 20.
    :param float beta: when policy_improvement_mode is "exp", this is the denominator
        of the exp function. Default to 1.
    :param float min_q_weight: weight for CQL loss/regularizer. Default to 10.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::
        Please refer to :class:`~tianshou.policy.PGPolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            critic: torch.nn.Module,
            optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            policy_improvement_mode: str = "exp",
            ratio_upper_bound: float = 20.0,
            beta: float = 1.0,
            min_q_weight: float = 10.0,
            target_update_freq: int = 0,
            reward_normalization: bool = False,
            state_tracker=None,
            buffer=None,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, critic, optim, discount_factor, policy_improvement_mode, ratio_upper_bound,
            beta, min_q_weight, target_update_freq, reward_normalization, **kwargs,
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
            **kwargs: Any,
    ) -> Batch:

        obs_emb = get_emb(self.state_tracker, buffer, indices=indices, obs=batch.obs, is_obs=is_obs)
        recommended_ids = get_recommended_ids(buffer) if remove_recommended_ids else None

        logits, hidden = self.actor(obs_emb, state=state)

        logits_masked, indices_masked = removed_recommended_id_from_embedding(logits, recommended_ids)

        if isinstance(logits_masked, tuple):
            dist = self.dist_fn(*logits_masked)
        else:
            dist = self.dist_fn(logits_masked)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits_masked.argmax(-1)
            elif self.action_type == "continuous":
                act = logits_masked[0]
        else:
            act = dist.sample()

        act_unsqueezed = act.unsqueeze(-1)
        act_ori = indices_masked.gather(dim=1, index=act_unsqueezed).squeeze(1)
        return Batch(logits=logits, act=act_ori, state=hidden, dist=dist)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:  # type: ignore
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()

        obs_emb = get_emb(self.state_tracker, self.buffer, batch.indices, is_obs=True)
        obs_next_emb = get_emb(self.state_tracker, self.buffer, batch.indices, is_obs=False)

        q_t = self.critic(obs_emb)
        # q_t = self.critic(batch.obs)
        act = to_torch(batch.act, dtype=torch.long, device=q_t.device)
        qa_t = q_t.gather(1, act.unsqueeze(1))
        # Critic loss
        with torch.no_grad():
            target_a_t, _ = self.actor_old(obs_next_emb)
            # target_a_t, _ = self.actor_old(batch.obs_next)
            target_m = Categorical(logits=target_a_t)
            q_t_target = self.critic_old(obs_next_emb)
            # q_t_target = self.critic_old(batch.obs_next)
            rew = to_torch_as(batch.rew, q_t_target)
            expected_target_q = (q_t_target * target_m.probs).sum(-1, keepdim=True)
            expected_target_q[batch.done > 0] = 0.0
            target = rew.unsqueeze(1) + self._gamma * expected_target_q
        critic_loss = 0.5 * F.mse_loss(qa_t, target)
        # Actor loss
        act_target, _ = self.actor(obs_emb)
        # act_target, _ = self.actor(batch.obs)
        dist = Categorical(logits=act_target)
        expected_policy_q = (q_t * dist.probs).sum(-1, keepdim=True)
        advantage = qa_t - expected_policy_q
        if self._policy_improvement_mode == "binary":
            actor_loss_coef = (advantage > 0).float()
        elif self._policy_improvement_mode == "exp":
            actor_loss_coef = (
                (advantage / self._beta).exp().clamp(0, self._ratio_upper_bound)
            )
        else:
            actor_loss_coef = 1.0  # effectively behavior cloning
        actor_loss = (-dist.log_prob(act) * actor_loss_coef).mean()
        # CQL loss/regularizer
        min_q_loss = (q_t.logsumexp(1) - qa_t).mean()
        loss = actor_loss + critic_loss + self._min_q_weight * min_q_loss
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {
            "loss": loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
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
