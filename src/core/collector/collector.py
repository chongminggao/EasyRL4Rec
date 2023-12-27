import gymnasium as gym
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
            self.env = DummyVectorEnv([lambda: env])  # type: ignore
        else:
            self.env = env  # type: ignore
        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise

        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = self.env.action_space
        # avoid creating attribute outside __init__
        self.reset() # Todo: whether use false

        self.remove_recommended_ids = remove_recommended_ids
        self.force_length = force_length

    

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

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset the environment, statistics, current data and possibly replay memory.

        :param bool reset_buffer: if true, reset the replay buffer that is attached
            to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(
            obs={},
            act={},
            rew={},
            rew_prev={},
            is_start={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={}
        )
        
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""

        self.buffer.reset(keep_statistics=keep_statistics)

        # ## Chongming
        # if batched_data is not None and batched_ids is not None:
        #     ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
        #         batched_data, buffer_ids=batched_ids
        #     )

    def reset_state_tracker(self, global_ids=None) -> None:
        """Reset the state tracker."""
        if self.preprocess_fn:
            self.preprocess_fn(dim_batch=self.env_num, reset=True, env_id=global_ids)


    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Reset all of the environments."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs, info = self.env.reset(**gym_reset_kwargs)
        # Note: here, obs is users' id

        self.data.obs = obs
        self.data.info = info
        self.data.rew_prev = np.zeros(len(obs))
        self.data.is_start = np.ones(len(obs), dtype=bool)


        # self.data.terminated = np.ones(self.env_num, dtype=bool)
        # self.data.truncated = np.zeros(self.env_num, dtype=bool)
        # self.data.rew = np.zeros(self.env_num)
        # self.data.obs = np.ones(self.env_num) * -1
        # self.data.act = np.ones(self.env_num) * -1
        # self.reset_buffer(batched_data=self.data, batched_ids=np.arange(self.env_num), keep_statistics=False)

        # self.reset_state_tracker()

        # initialize the statetracker with the user's id.
        # if self.preprocess_fn:
        #     processed_data = self.preprocess_fn(
        #         obs=obs, info=info, env_id=np.arange(self.env_num)
        #         )
        #     obs = processed_data.get("obs", obs)
        #     info = processed_data.get("info", info)
        # self.data.info = info
        # self.data.obs = obs

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

    def _reset_env_with_ids(
        self,
        local_ids: Union[List[int], np.ndarray],
        global_ids: Union[List[int], np.ndarray],
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs_reset, info = self.env.reset(global_ids, **gym_reset_kwargs)


        self.data.obs[local_ids] = obs_reset
        self.data.info[local_ids] = info
        self.data.rew_prev[local_ids] = np.zeros(len(obs_reset))
        self.data.is_start[local_ids] = np.ones(len(obs_reset), dtype=bool)
        
        # # Note: here, obs_reset is users' id
        # batched_data = Batch(
        #     obs_next = obs_reset,
        #     info = info,
        #     terminated = np.ones(len(global_ids), dtype=bool),
        #     truncated = np.zeros(len(global_ids), dtype=bool),
        #     rew = np.zeros(len(global_ids)),
        #     obs = np.ones(self.env_num) * -1,
        #     act = np.ones(self.env_num) * -1
        # )
        # self.reset_buffer(batched_data=batched_data, batched_ids=global_ids, keep_statistics=True)

        # self.reset_state_tracker(global_ids)

        # if self.preprocess_fn:
        #     processed_data = self.preprocess_fn(
        #         obs=obs_reset, info=info, env_id=global_ids
        #     )
        #     obs_reset = processed_data.get("obs", obs_reset)
        #     info = processed_data.get("info", info)
        
        
        # info = [{"cum_reward": item["cum_reward"], "env_id": value} for item, value in zip(info, global_ids)]
        # self.data.info[local_ids] = info
        
        # self.data.obs_next[local_ids] = obs_reset
        # self.data[local_ids] = batched_data

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
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
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
            * ``rew`` mean of episodic rewards.
            * ``len`` mean of episodic lengths.
            * ``rew_std`` standard error of episodic rewards.
            * ``len_std`` standard error of episodic lengths.
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
            self.data = self.data[:min(self.env_num, n_episode)]

        # self.data = self.data[:min(self.env_num, n_episode)]
        # self.reset()  # Instead of using the last obs, we generate new obs using updated parameters.
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
                )
    
        # self.reset() # chongming added

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
                try:
                    act_sample = [
                        self._action_space[i].sample() for i in ready_env_ids
                    ]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore 
                self.data.update(act=act_sample)
            else:
                
                indices = self.buffer.last_index if len(self.buffer) > 0 else None
                if indices is not None:
                    indices = indices[ready_env_ids] if n_episode else indices

                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, self.buffer, indices=indices, is_obs=True, state=last_state, remove_recommended_ids=self.remove_recommended_ids, is_train=is_train, use_batch_in_statetracker=True)
                else:
                    result = self.policy(self.data, self.buffer, indices=indices, is_obs=True, state=last_state, remove_recommended_ids=self.remove_recommended_ids, is_train=is_train, use_batch_in_statetracker=True)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())  # Todo: 这里在pg下是空的！
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise and hasattr(self.policy.policy, "eps") and self.policy.action_type == 'discrete':  # Policy-based methods (e.g., A2C) has no eps.
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # add noise only in train env
            self.policy.remap_exploration_noise = self.exploration_noise
            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data)  # RecPolicy transform!

            obs_next, rew, terminated, truncated, info = self.env.step(action_remap, ready_env_ids)

            cnt_loop += 1
            if self.force_length > 0:
                if cnt_loop >= self.force_length:
                    terminated = np.ones_like(terminated, dtype=bool)
                else:
                    terminated = np.zeros_like(terminated, dtype=bool)
            
            done = np.logical_or(terminated, truncated)
            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                    obs_next=self.data.obs_next,
                    rew=self.data.rew,
                    terminated=self.data.terminated,
                    truncated=self.data.truncated,
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
                self.data, buffer_ids=ready_env_ids
                )
            self.data.is_start = np.zeros(len(self.data), dtype=bool)

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

                
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

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

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env() # The reset will be run after parameter update!!

        if episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
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
