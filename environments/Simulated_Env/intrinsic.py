import numpy as np
import torch

from environments.Simulated_Env.base import BaseSimulatedEnv

# from virtualTB.model.UserModel import UserModel
# from environments.VirtualTaobao.virtualTB.utils import *


class IntrinsicSimulatedEnv(BaseSimulatedEnv):
    def __init__(self, ensemble_models,
                 env_task_class, task_env_param: dict, task_name: str,
                 predicted_mat=None,
                 item_similarity=None,
                 item_popularity=None,
                 lambda_diversity=0.1,
                 lambda_novelty=0.1,
                 ):
        super().__init__(ensemble_models, env_task_class, task_env_param, task_name, predicted_mat)
        self.item_similarity = item_similarity
        self.item_popularity = item_popularity
        self.lambda_diversity = lambda_diversity
        self.lambda_novelty = lambda_novelty

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
        else:  # elif self.env_name == "KuaiEnv-v0":
            # get prediction
            pred_reward = self.predicted_mat[self.cur_user, action]  # todo
        
        # get diversity
        div_reward = self._cal_diversity(action)
        # get novelty
        nov_reward = self._cal_novelty(action)
        intrinsic_reward = self.lambda_diversity * div_reward + \
                           self.lambda_novelty * nov_reward
        final_reward = pred_reward + intrinsic_reward - self.MIN_R
        return final_reward

    def _cal_diversity(self, action):
        l = len(self.history_action)
        div = 0.0
        if l <= 1:
            return 0.0
        for i in range(l-1):
            div += (1 - self.item_similarity[self.history_action[i], action])
        div /= (l-1)
        return div
    
    def _cal_novelty(self, action):
        if hasattr(self.env_task, "lbe_item"):        
            p_id = self.env_task.lbe_item.inverse_transform([action])[0]
        else:
            p_id = action
        item_pop = self.item_popularity[p_id]
        nov = - np.log(item_pop+1e-10)  # nov \in xxxx   -log(0.0001) = 4  lambda 考虑~0.25
        return nov
