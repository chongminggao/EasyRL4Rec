from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from core.policy.utils import get_emb, get_recommended_ids, removed_recommended_id_from_embedding
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import DQNPolicy


class DQNPolicy_with_Embedding(DQNPolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double dqn. Default to True.
    :param bool clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss. Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        state_tracker,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        **kwargs: Any,

    ) -> None:
        super().__init__(model, optim, discount_factor=discount_factor, estimation_step=estimation_step,
                         target_update_freq=target_update_freq, reward_normalization=reward_normalization,
                         is_double=is_double, clip_loss_grad=clip_loss_grad, **kwargs)
        self.state_tracker = state_tracker
    
    def set_collector(self, train_collector):  ## TODO
        self.train_collector = train_collector

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        # result = self(batch, input="obs_next")  ## TODO
        result = self(batch, buffer, indices, is_obs=False, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            # target_q = self(batch, model="model_old", input="obs_next").logits
            target_q = self(batch, buffer, indices, is_obs=False, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]




    def forward(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray = None,
        is_obs=None,
        remove_recommended_ids=False,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        # input: str = "obs",
        use_batch_in_statetracker=False,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        # obs = batch[input]
        # obs_next = obs.obs if hasattr(obs, "obs") else obs
        # is_obs = True if input == "obs" else False  ## TODO
        # assert not hasattr(batch.obs, "obs")
        # print(batch.obs)
        # print(indices)
        # print(buffer)
        
        obs_emb = get_emb(self.state_tracker, buffer, indices=indices, batch=batch, is_obs=is_obs, use_batch_in_statetracker=use_batch_in_statetracker)  ## TODO is_train=is_train
        # print(obs_emb)
        # assert False
        recommended_ids = get_recommended_ids(buffer) if remove_recommended_ids else None

        logits, hidden = model(obs_emb, state=state, info=batch.info)

        q = self.compute_q_value(logits, getattr(obs_emb, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]

        logits_masked, indices_masked = removed_recommended_id_from_embedding(q, recommended_ids)
        # act = to_numpy(q.max(dim=1)[1])
        act = logits_masked.max(dim=1)[1]

        act_unsqueezed = act.unsqueeze(-1)
        act_ori = indices_masked.gather(dim=1, index=act_unsqueezed).squeeze(1)

        return Batch(logits=logits, act=act_ori, state=hidden)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()

        optim_RL, optim_state = self.optim  ## TODO
        optim_RL.zero_grad()
        optim_state.zero_grad()

        weight = batch.pop("weight", 1.0)
        # q = self(batch).logits
        q = self(batch, self.train_collector.buffer, indices=batch.indices, is_obs=True).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        if self._clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        optim_RL.step()
        optim_state.step()

        self._iter += 1
        return {"loss": loss.item()}

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
