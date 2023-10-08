from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
import random
import torch

class BaseEnv(ABC, gym.Env):
    def __init__(self, num_leave_compute, leave_threshold, max_turn, random_init):
        
        self.max_turn = max_turn
        self.random_init = random_init
        
        self.observation_space = gym.spaces.Box(low=0, high=len(self.mat) - 1, shape=(1,), dtype=np.int32)
        self.action_space = gym.spaces.Box(low=0, high=self.mat.shape[1] - 1, shape=(1,), dtype=np.int32)

        self.num_leave_compute = num_leave_compute
        self.leave_threshold = leave_threshold

        self.reset()

    @property
    def state(self):
        state = np.array([self.cur_user, self.action])
        return state
    
    def __user_generator(self):
        user = random.randint(0, len(self.mat) - 1)
        return user
    
    def __item_generator(self):
        if self.random_init:
            item = random.randint(0, len(self.mat[0]) - 1)
        else:
            item = len(self.mat[0])
        return item

    def step(self, action):
        self.action = action
        t = self.total_turn
        terminated = self._determine_whether_to_leave(t, action)
        if t >= (self.max_turn - 1):
            terminated = True
        self._add_action_to_history(t, action)

        reward = self.mat[self.cur_user, action]

        self.cum_reward += reward
        self.total_turn += 1

        truncated = False

        return self.state, reward, terminated, truncated, {'cum_reward': self.cum_reward}

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.cur_user = self.__user_generator()
        self.action = self.__item_generator()
        self._reset_history()

        # return self.state, {'key': 1, 'env': self}
        return self.state, {'cum_reward': 0.0}
    

    def _reset_history(self):
        self.history_action = {}
        self.sequence_action = []
        self.max_history = 0

    def _add_action_to_history(self, t, action):
        self.sequence_action.append(action)
        self.history_action[t] = action
        assert self.max_history == t
        self.max_history += 1

    def seed(self, sd=0):
            torch.manual_seed(sd)
