import numpy as np
import torch

from environments.Simulated_Env.base import BaseSimulatedEnv
from torch import FloatTensor
from src.core.util.utils import compute_action_distance, clip0, compute_exposure


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
                 entropy_min=0,
                 entropy_max=0,
                 feature_level=False,
                 map_item_feat=None,
                 is_sorted=True
                 ):
        super().__init__(ensemble_models, env_task_class, task_env_param, task_name, predicted_mat)
        self.feature_level = feature_level
        self.is_sorted = is_sorted
        self.map_item_feat = map_item_feat
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

    def _compute_pred_reward(self, action):
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
            # action_reverse = action_trans[::-1]
            # max_win = max(self.entropy_window)
            for k in entropy_set:
                if len(action_trans) < k:
                    entropy += 1 # todo! 补足差额
                    continue

                action_set = tuple(sorted(action_trans[-k:])) if self.is_sorted else tuple(action_trans[-k:])
                # print(action_set)
                if self.feature_level:
                    feat_set = get_features_of_last_n_items_features(k, action_set, self.map_item_feat, is_sort=self.is_sorted)
                    if len(feat_set) == 0:
                        entropy += 1
                        continue
                    ans_feat = 0
                    for feat in feat_set:
                        if feat in self.entropy_dict["map"]:
                            ans_feat += self.entropy_dict["map"][feat]
                        else:
                            ans_feat += 1 # todo! 补足差额
                    entropy += ans_feat / len(feat_set)
                else:
                    if action_set in self.entropy_dict["map"]:
                        entropy += self.entropy_dict["map"][action_set]
                    else:
                        entropy += 1  # todo! 补足差额

            # if len(action_trans) < self.step_n_actions:
            #     entropy += self.step_n_actions - len(action_trans)  # todo 补足差额

        penalized_reward = pred_reward + self.lambda_entropy * entropy - self.MIN_R

        ##############################

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

        final_reward = max(0, final_reward)
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


def get_features_of_last_n_items_features(n, hist_tra, map_item_feat, is_sort):
    if len(hist_tra) < n or n <= 0:
        return [[]]
    target_item = hist_tra[-1]
    target_features = map_item_feat[target_item]
    last_lists = get_features_of_last_n_items_features(n - 1, hist_tra[:-1], map_item_feat, is_sort)
    res = set()
    for feat_list in last_lists:
        feat_list = list(feat_list)
        for feat in target_features:
            new_list = feat_list.copy()
            new_list.append(feat)
            if is_sort:
                new_list = sorted(new_list)
            new_tuple = tuple(new_list)
            res.add(new_tuple)
    return res