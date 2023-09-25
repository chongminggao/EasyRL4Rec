import numpy as np
import torch

from environments.Simulated_Env.base import BaseSimulatedEnv
from torch import FloatTensor
from core.util.utils import compute_action_distance, clip0, compute_exposure

# from virtualTB.model.UserModel import UserModel
# from environments.VirtualTaobao.virtualTB.utils import *

class PenaltyEntExpSimulatedEnv(BaseSimulatedEnv):
    def __init__(self, ensemble_models,
                 env_task_class, task_env_param: dict, task_name: str,
                 predicted_mat=None, 
                 version: str = "v1", tau: float = 1.0,
                 use_exposure_intervention=False,
                 gamma_exposure=1,
                 alpha_u=None, beta_i=None,
                 entropy_dict=None,
                 entropy_window=None,
                 lambda_entropy=1,
                 step_n_actions=1,
                 entropy_min=None,
                 entropy_max=None,
                 ):
        super().__init__(ensemble_models, env_task_class, task_env_param, task_name, predicted_mat)
        self.version = version
        self.tau = tau
        self.use_exposure_intervention = use_exposure_intervention
        self.alpha_u = alpha_u
        self.beta_i = beta_i
        self.gamma_exposure = gamma_exposure
        self.entropy_dict = entropy_dict
        self.entropy_window = entropy_window
        self.step_n_actions = step_n_actions
        self.lambda_entropy = lambda_entropy

        self.MIN_R = predicted_mat.min() + lambda_entropy * entropy_min
        self.MAX_R = predicted_mat.max() + lambda_entropy * entropy_max

        self._reset_history_exposure()

    def _compute_pred_reward(self,action):
        if self.env_name == "VirtualTB-v0":
            feature = np.concatenate((self.cur_user, np.array([self.reward, 0, self.total_turn]), action), axis=-1)
            feature_tensor = torch.unsqueeze(torch.tensor(feature, device=self.user_model.device, dtype=torch.float), 0)
            # pred_reward = self.user_model(feature_tensor).detach().cpu().numpy().squeeze().round()
            pred_reward = self.user_model.forward(feature_tensor).detach().cpu().numpy().squeeze()
            if pred_reward < 0:
                pred_reward = 0
            if pred_reward > 10:
                pred_reward = 10
            penalized_reward = pred_reward
        else:  # elif self.env_name == "KuaiEnv-v0":
            # 1. get prediction
            pred_reward = self.predicted_mat[self.cur_user, action]  # todo
            # 2. get entropy
            # entropy_u = 0
            # if 0 in self.entropy_window:
            #     entropy_u = self.entropy_dict["on_user"].loc[self.cur_user]
            entropy = 0
            entropy_set = set(self.entropy_window) - {0}
            if len(entropy_set):
                action_k = self.history_action[max(0, self.total_turn - self.step_n_actions + 1):self.total_turn + 1]
                if hasattr(self.env_task, "lbe_item") and self.env_task.lbe_item:
                    action_trans = self.env_task.lbe_item.inverse_transform(action_k)
                else:
                    action_trans = action_k
                action_reverse = action_trans[::-1]
                for k in range(len(action_reverse)):
                    action_set = tuple(sorted(action_reverse[:k + 1]))
                    # print(action_set)
                    if action_set in self.entropy_dict["map"]:
                        entropy += self.entropy_dict["map"][action_set]
                    else:
                        entropy += 1 # todo! 补足差额
                if len(action_reverse) < self.step_n_actions:
                    entropy += self.step_n_actions - len(action_reverse) # todo 补足差额
            penalized_reward = pred_reward + self.lambda_entropy * entropy - self.MIN_R

        # Compute intervened exposure effect e^*_t(u, i)
        t = int(self.total_turn)
        if self.use_exposure_intervention:
            exposure_effect = self._compute_exposure_effect(t, action)
        else:
            exposure_effect = 0
        if t < self.env_task.max_turn:
            self._add_exposure_to_history(t, exposure_effect)

        if self.version == "v1":  # version 1
            final_reward = clip0(penalized_reward) / (1.0 + exposure_effect)
        else:  # version 2
            final_reward = clip0(penalized_reward - exposure_effect)

        return final_reward
    
    def _compute_exposure_effect(self, t, action):
        if t == 0:
            return 0
        a_history = self.history_action[:t]
        distance = compute_action_distance(action, a_history, self.env_name, self.env_task)
        t_diff = t - np.arange(t)
        exposure_effect = compute_exposure(t_diff, distance, self.tau)

        if self.alpha_u is not None:
            u_id = self.env_task.lbe_user.inverse_transform(self.cur_user)[0]
            p_id = self.env_task.lbe_item.inverse_transform([action])[0]
            a_u = self.alpha_u[u_id]
            b_i = self.beta_i[p_id]
            exposure_effect_new = float(exposure_effect * a_u * b_i)
        else:
            exposure_effect_new = exposure_effect

        exposure_gamma = exposure_effect_new * self.gamma_exposure
        return exposure_gamma

    def _reset_history_exposure(self):
        self.history_exposure = {}

    def _add_exposure_to_history(self, t, exposure):
        self.history_exposure[t] = exposure
