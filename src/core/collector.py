import gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.policy import BasePolicy
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import (
    Batch,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    CachedReplayBuffer,
    to_numpy,
)


class Collector(object):
    def __init__(
            self,
            policy: BasePolicy,
            env: Union[gym.Env, BaseVectorEnv],
            buffer: Optional[ReplayBuffer] = None,
            preprocess_fn: Optional[Callable[..., Batch]] = None,
            exploration_noise: bool = False,
            remove_recommended_ids=False,
            force_length=0,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to DummyVectorEnv.")
            env = DummyVectorEnv([lambda: env])
        self.env = env
        self.env_num = len(env)
        self.exploration_noise = exploration_noise
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = env.action_space
        # avoid creating attribute outside __init__
        self.reset()

        self.remove_recommended_ids = remove_recommended_ids
        self.force_length = force_length

    """Revised tianshou.data.collector class.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    The "preprocess_fn" is a function called before the data has been added to the
    buffer with batch format. It will receive only "obs" and "env_id" when the
    collector resets the environment, and will receive six keys "obs_next", "rew",
    "done", "info", "policy" and "env_id" in a normal env step. It returns either a
    dict or a :class:`~tianshou.data.Batch` with the modified keys and values. Examples
    are in "test/base/test_collector.py".

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.
    """

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if type(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def reset(self) -> None:
        """Reset all related variables in the collector."""
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(obs={}, act={}, rew={}, done={},
                          obs_next={}, info={}, policy={})
        self.reset_env()
        self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""

        ## Chongming
        # maxsize = self.buffer.maxsize
        # buffer_num = self.buffer.buffer_num
        # buffer = VectorReplayBuffer(maxsize, buffer_num)
        # self._assign_buffer(buffer)

        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(self) -> None:
        """Reset all of the environments."""

        # Chongming
        if self.preprocess_fn:
            self.preprocess_fn(dim_batch=self.env_num, reset=True)

        obs = self.env.reset()
        if self.preprocess_fn:
            obs = self.preprocess_fn(
                obs=obs, env_id=np.arange(self.env_num)).get("obs", obs)
        self.data.obs = obs

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def collect(
            self,
            n_step: Optional[int] = None,
            n_episode: Optional[int] = None,
            random: bool = False,
            render: Optional[float] = None,
            no_grad: bool = True,
            is_train=True,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode. Revised from tianshou.data.collector

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))

            # self.data = self.data[:min(self.env_num, n_episode)]
            self.reset()  # Instead of using the last obs, we generate new obs using updated parameters.

        else:
            raise TypeError("Please specify at least one (either n_step or n_episode) "
                            "in AsyncCollector.collect().")

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        if render:
            render_list = []


        cnt_loop = 0
        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)  # Todo: 这里在pg下是空的！

            # get the next action
            if random:
                self.data.update(
                    act=[self._action_space[i].sample() for i in ready_env_ids])
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, self.buffer, state=last_state, remove_recommended_ids=self.remove_recommended_ids, is_train=is_train)
                else:
                    result = self.policy(self.data, self.buffer, state=last_state, remove_recommended_ids=self.remove_recommended_ids, is_train=is_train)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())  # Todo: 这里在pg下是空的！
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)

            # if len(self.buffer) and self.step_n_actions > 1:
            #     now_ind = self.buffer.last_index[ready_env_ids]
            #     action_k = np.vstack([action_remap, self.buffer[now_ind].act])
            #
            #     for i in range(self.step_n_actions-2):
            #         prev_ind = self.buffer.prev(now_ind)
            #         if all(prev_ind == now_ind):
            #             break
            #         assert all(prev_ind != now_ind)
            #         now_ind = prev_ind
            #         action_k = np.vstack([action_k, self.buffer[now_ind].act])
            #     action_k = action_k.T
            #
            # else:
            #     action_k = action_remap
            #
            # obs_next, rew, done, info = self.env.step(action_k, ready_env_ids) # type: ignore

            # self.env.get_env_attr("cur_user")

            obs_next, rew, done, info = self.env.step(
                action_remap, ready_env_ids)

            cnt_loop += 1
            if self.force_length > 0:
                if cnt_loop >= self.force_length:
                    done = np.ones_like(done, dtype=bool)
                else:
                    done = np.zeros_like(done, dtype=bool)

            self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)
            if self.preprocess_fn:
                self.data.update(self.preprocess_fn(
                    obs_next=self.data.obs_next,
                    rew=self.data.rew,
                    done=self.data.done,
                    info=self.data.info,
                    policy=self.data.policy,
                    env_id=ready_env_ids,
                ))

            if render:
                render_result = self.env.render()
                render_list.append(render_result)
                # if render > 0 and not np.isclose(render, 0):
                #     time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids)

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.

                # # Delete for efficiency
                # obs_reset = self.env.reset(env_ind_global)
                # if self.preprocess_fn:
                #     obs_reset = self.preprocess_fn(
                #         obs=obs_reset, env_id=env_ind_global).get("obs", obs_reset)
                # self.data.obs_next[env_ind_local] = obs_reset
                # for i in env_ind_local:
                #     self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next  # Todo: 注意这里的状态更新

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        # if n_episode:
        #     self.data = Batch(obs={}, act={}, rew={}, done={},
        #                       obs_next={}, info={}, policy={})
        #     self.reset_env() # The reset will be run after parameter update!!

        if episode_count > 0:
            rews, lens, idxs = list(map(
                np.concatenate, [episode_rews, episode_lens, episode_start_indices]))
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        res = {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }

        if render:
            return render_list, res

        return res
