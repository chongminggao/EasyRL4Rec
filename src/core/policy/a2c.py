import pprint
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.policy.utils import get_emb, removed_recommended_id_from_embedding, get_recommended_ids
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic


class A2CPolicy_withEmbedding(A2CPolicy):
    """Implementation of Synchronous Advantage Actor-Critic. arXiv:1602.01783.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close to
        1. Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.
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
            optim: torch.optim.Optimizer,
            dist_fn: Type[torch.distributions.Distribution],
            state_tracker,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            max_grad_norm: Optional[float] = None,
            gae_lambda: float = 0.95,
            max_batchsize: int = 256,
            **kwargs: Any
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, vf_coef=vf_coef, ent_coef=ent_coef, max_grad_norm=max_grad_norm,
                         gae_lambda=gae_lambda, max_batchsize=max_batchsize, **kwargs)
        self.state_tracker = state_tracker

    def set_collector(self, train_collector):
        self.train_collector = train_collector

    # def process_fn(
    #     self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    # ) -> Batch:
    #     batch = self._compute_returns(batch, buffer, indices)
    #     batch.act = to_torch_as(batch.act, batch.v_s)
    #     return batch

    def _compute_returns(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            batch.indices = indices
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                obs_emb = get_emb(self.state_tracker, buffer, minibatch.indices, is_obs=True)
                # v_s.append(self.critic(minibatch.obs))
                v_s.append(self.critic(obs_emb))
                obs_next_emb = get_emb(self.state_tracker, buffer, minibatch.indices, is_obs=False)
                # v_s_.append(self.critic(minibatch.obs_next))
                v_s_.append(self.critic(obs_next_emb))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                            np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

    def forward(
            self,
            batch: Batch,
            buffer: ReplayBuffer,
            indices: np.ndarray = None,
            is_obs=None,
            remove_recommended_ids=False,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            is_train=True,
            use_batch_in_statetracker=False,
            **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """

        obs_emb = get_emb(self.state_tracker, buffer, indices=indices, is_obs=is_obs, batch=batch, is_train=is_train, use_batch_in_statetracker=use_batch_in_statetracker)
        recommended_ids = get_recommended_ids(buffer) if remove_recommended_ids else None

        logits, hidden = self.actor(obs_emb, state=state)

        logits_masked, indices_masked = removed_recommended_id_from_embedding(logits, recommended_ids)

        if isinstance(logits_masked, tuple):
            dist = self.dist_fn(*logits_masked)
        else:
            try:
                dist = self.dist_fn(logits_masked)
            except:
                logits_masked_cpu = logits_masked.cpu()
                import datetime
                import time
                import pprint
                nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
                torch.save(logits_masked_cpu, f"logits_masked_cpu_{nowtime}.pt")
                print(logits_masked)
                print(logits_masked.sum(1))
                print(logits_masked[logits_masked<0])
                dist = self.dist_fn(logits_masked+1)
                # a = torch.load("logits_masked_cpu_2023_01_15-17_08_28.pt")
                # dist = self.dist_fn(a)

        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits_masked.argmax(-1)
            elif self.action_type == "continuous":
                act = logits_masked[0]
        else:
            act = dist.sample()

        act_unsqueezed = act.unsqueeze(-1)
        act_ori = indices_masked.gather(dim=1, index=act_unsqueezed).squeeze(1)

        # if isinstance(logits, tuple):
        #     dist = self.dist_fn(*logits)
        # else:
        #     dist = self.dist_fn(logits)
        # if self._deterministic_eval and not self.training:
        #     if self.action_type == "discrete":
        #         act = logits.argmax(-1)
        #     elif self.action_type == "continuous":
        #         act = logits[0]
        # else:
        #     act = dist.sample()

        return Batch(logits=logits, act=act_ori, state=hidden, dist=dist)

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []

        optim_RL, optim_state = self.optim
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor

                # dist = self(minibatch).dist
                dist = self.forward(minibatch, self.train_collector.buffer, indices=minibatch.indices, is_obs=True).dist

                log_prob = dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()
                # calculate loss for critic

                obs_emb = get_emb(self.state_tracker, self.train_collector.buffer, minibatch.indices, is_obs=True)
                value = self.critic(obs_emb).flatten()
                vf_loss = F.mse_loss(minibatch.returns, value)
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = actor_loss + self._weight_vf * vf_loss \
                       - self._weight_ent * ent_loss
                optim_RL.zero_grad()
                optim_state.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                optim_RL.step()
                optim_state.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def exploration_noise(
            self,
            act: Union[np.ndarray, Batch],
            batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.actor.output_dim)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act
