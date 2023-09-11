import math
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from core.policy.utils import get_emb, get_recommended_ids, removed_recommended_id_from_embedding
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import DiscreteBCQPolicy


class SQN(DiscreteBCQPolicy):
    """Implementation of discrete BCQ algorithm. arXiv:1910.01708.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> q_value)
    :param torch.nn.Module imitator_final_layer: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> imitation_logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency.
    :param float eval_eps: the epsilon-greedy noise added in evaluation.
    :param float unlikely_action_threshold: the threshold (tau) for unlikely
        actions, as shown in Equ. (17) in the paper. Default to 0.3.
    :param float imitation_logits_penalty: regularization weight for imitation
        logits. Default to 1e-2.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            model_final_layer: torch.nn.Module,
            imitator_final_layer: torch.nn.Module,
            optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            estimation_step: int = 1,
            target_update_freq: int = 8000,
            eval_eps: float = 1e-3,
            unlikely_action_threshold: float = 0.3,
            imitation_logits_penalty: float = 1e-2,
            reward_normalization: bool = False,
            state_tracker=None,
            buffer=None,
            which_head="shead",
            **kwargs: Any,
    ) -> None:
        super().__init__(
            model_final_layer, imitator_final_layer, optim, discount_factor, estimation_step, target_update_freq,
            eval_eps, unlikely_action_threshold, imitation_logits_penalty, reward_normalization, **kwargs
        )
        self.state_tracker = state_tracker
        self.buffer = buffer
        assert which_head in {"shead", "qhead", "bcq"}
        self.which_head = which_head

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
        act = self(batch, buffer, indices=indices, is_obs=False, input="obs_next").act

        # obs = batch["obs_next"]
        obs_emb = get_emb(self.state_tracker, buffer, indices=indices, is_obs=False, batch=batch)
        target_q, _ = self.model_old(obs_emb)

        # target_q, _ = self.model_old(batch.obs_next)
        target_q = target_q[np.arange(len(act)), act]
        return target_q

    def forward(  # type: ignore
            self,
            batch: Batch,
            buffer: ReplayBuffer,
            indices: np.ndarray = None,
            is_obs=None,
            remove_recommended_ids=False,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            use_batch_in_statetracker=False,
            **kwargs: Any,
    ) -> Batch:
        # assert input == "obs"
        is_obs = True if input == "obs" else False
        # obs = batch[input]
        obs_emb = get_emb(self.state_tracker, buffer, indices=indices, is_obs=is_obs, batch=batch, use_batch_in_statetracker=use_batch_in_statetracker)
        recommended_ids = get_recommended_ids(buffer) if remove_recommended_ids else None

        q_value, state = self.model(obs_emb, state=state, info=batch.info)
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q_value.shape[1]
        imitation_logits, _ = self.imitator(obs_emb, state=state, info=batch.info)

        q_value_masked, indices_masked = removed_recommended_id_from_embedding(q_value, recommended_ids)
        imitation_logits_masked, indices_masked = removed_recommended_id_from_embedding(imitation_logits, recommended_ids)


        if self.which_head == "bcq":
            # BCQ way
            ratio = imitation_logits_masked - imitation_logits_masked.max(dim=-1, keepdim=True).values
            mask = (ratio < self._log_tau).float()
            act = (q_value_masked - np.inf * mask).argmax(dim=-1)
        elif self.which_head == "shead":
            # Supervised head
            act = imitation_logits_masked.argmax(dim=-1)
        elif self.which_head == "qhead":
            act = q_value_masked.argmax(dim=-1)

        act_unsqueezed = act.unsqueeze(-1)
        act_ori = indices_masked.gather(dim=1, index=act_unsqueezed).squeeze(1)

        return Batch(
            act=act_ori, state=state, q_value=q_value, imitation_logits=imitation_logits
        )

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._iter % self._freq == 0:
            self.sync_weight()
        self._iter += 1

        target_q = batch.returns.flatten()
        result = self.forward(batch, self.buffer, batch.indices, is_obs=True, input="obs")
        # result = self(batch)
        imitation_logits = result.imitation_logits
        current_q = result.q_value[np.arange(len(target_q)), batch.act]
        act = to_torch(batch.act, dtype=torch.long, device=target_q.device)
        q_loss = F.smooth_l1_loss(current_q, target_q)
        i_loss = F.nll_loss(F.log_softmax(imitation_logits, dim=-1), act)
        reg_loss = imitation_logits.pow(2).mean()
        loss = q_loss + i_loss + self._weight_reg * reg_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            "loss": loss.item(),
            "loss/q": q_loss.item(),
            "loss/i": i_loss.item(),
            "loss/reg": reg_loss.item(),
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

    # def process_fn(
    #     self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    # ) -> Batch:
    #     """Compute the n-step return for Q-learning targets.
    #
    #     More details can be found at
    #     :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
    #     """
    #     batch = self.compute_nstep_return(
    #         batch, buffer, indices, self._target_q, self._gamma, self._n_step,
    #         self._rew_norm
    #     )
    #     batch.indices = indices
    #     return batch
