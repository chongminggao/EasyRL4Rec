import math

import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Type, Optional, Union

from tianshou.policy import A2CPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as


class PPOPolicy(A2CPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            critic: torch.nn.Module,
            optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
            dist_fn: Type[torch.distributions.Distribution],
            eps_clip: float = 0.2,
            dual_clip: Optional[float] = None,
            value_clip: bool = False,
            advantage_normalization: bool = True,
            recompute_advantage: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        if not self._rew_norm:
            assert not self._value_clip, \
                "value clip is available only when `reward_normalization` is True"
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indice` to be used in `learn()`.
            self._buffer, self._indice = buffer, indice
        batch = self._compute_returns(batch, buffer, indice)
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(b).dist.log_prob(b.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        return batch

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []

        optim_RL, optim_state = self.optim

        for step in range(repeat):
            optim_state.zero_grad()
#             if step == repeat - 1:
#                 optim_state.zero_grad()
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indice)

            batches = batch.split(batch_size, merge_last=True)
            for b_ind, b in enumerate(batches):
                # calculate loss for actor
                dist = self(b).dist
                if self._norm_adv:
                    mean, std = b.adv.mean(), b.adv.std()
                    b.adv = (b.adv - mean) / std  # per-batch norm
                ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(
                        torch.min(surr1, surr2), self._dual_clip * b.adv
                    ).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                # calculate loss for critic
                value = self.critic(b.obs).flatten()
                if self._value_clip:
                    v_clip = b.v_s + (value - b.v_s).clamp(
                        -self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (b.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()

                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss

                optim_RL.zero_grad()
                loss.backward(retain_graph=True)
                # if step == repeat - 1 and b_ind == math.floor(batch.__len__()/batch_size) - 1:
                #     loss.backward()
                # else:
                #     loss.backward(retain_graph=True)

                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        max_norm=self._grad_norm)

                optim_RL.step()
                # if step == repeat - 1 and b_ind == math.floor(batch.__len__()/batch_size) - 1:
                #     optim_state.step()

                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        optim_state.step() # Only update at the last one batch.

        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
