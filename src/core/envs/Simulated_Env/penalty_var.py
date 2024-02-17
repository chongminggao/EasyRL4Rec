import numpy as np
import torch

from src.core.envs.Simulated_Env.base import BaseSimulatedEnv

# from virtualTB.model.UserModel import UserModel
# from src.core.envs.VirtualTaobao.virtualTB.utils import *


class PenaltyVarSimulatedEnv(BaseSimulatedEnv):
    def __init__(self, ensemble_models,
                 env_task_class, task_env_param: dict, task_name: str,
                 predicted_mat=None, 
                 maxvar_mat=None,
                 lambda_variance=1,
                 ):
        super().__init__(ensemble_models, env_task_class, task_env_param, task_name, predicted_mat)
        self.maxvar_mat = maxvar_mat
        self.lambda_variance = lambda_variance
        self.MIN_R = predicted_mat.min() - lambda_variance * maxvar_mat.max()
        self.MAX_R = predicted_mat.max() - lambda_variance * maxvar_mat.min()
        
    def _compute_pred_reward(self, action):
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
            # 2. get variance
            max_var = self.maxvar_mat[self.cur_user, action]  # todo

        penalized_reward = pred_reward - self.lambda_variance * max_var - self.MIN_R    
        return penalized_reward
