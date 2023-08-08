# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 3:01 下午
# @Author  : Chongming GAO
# @FileName: linucb.py
import numpy as np
from tqdm import tqdm

from core.evaluation.evaluator_static import test_static_model_in_RL_env


class linucb_disjoint_arm():

    def __init__(self, arm_index, d, alpha):
        # Track arm index
        self.arm_index = arm_index

        # Keep track of alpha
        self.alpha = alpha

        # A: (d x d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(d)

        # b: (d x 1) corresponding response vector.
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d, 1])

    def calc_reward(self, x_array):
        x = x_array.reshape([-1, 1])
        reward = np.dot(self.theta.T, x)
        return reward

    @property
    def A_inv(self):
        return np.linalg.inv(self.A)

    @property
    def theta(self):
        theta = np.dot(self.A_inv, self.b)
        return theta

    def calc_UCB(self, x_array):
        # Find A inverse for ridge regression
        # A_inv = np.linalg.inv(self.A)

        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        # self.theta = np.dot(A_inv, self.b)

        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1, 1])

        # Find ucb based on p formulation (mean + std_dev)
        # p is (1 x 1) dimension vector
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(self.A_inv, x)))

        return p

    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1, 1])

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)

        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x


class linucb_policy():

    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index=i, d=d, alpha=alpha) for i in range(K_arms)]

    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -1

        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []

        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)

            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                # Set new max ucb
                highest_ucb = arm_ucb

                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)

        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)

        return chosen_arm

    def forward(self, arm, x_array):
        reward = self.linucb_arms[arm].calc_reward(x_array)
        return reward

    def evaluate_data(self, dataset_val, metric_fun, lbe_video):
        y = dataset_val.get_y()
        # y_predict = self.predict_data(dataset_val, batch_size)
        x_val = dataset_val.x_numpy

        y_predict = np.zeros_like(y)

        # x_val = x_val[:10000, :]
        # y = y[:10000]
        # y_predict = y_predict[:10000]

        for i in tqdm(range(len(x_val)), desc="Predicting results"):
            x = x_val[i, :]
            arm = int(x[1])
            if not arm in lbe_video.classes_:
                continue
            arm_small = lbe_video.transform([arm])[0]
            y_predict[i] = self.forward(arm_small, x)

        eval_result = {}
        for name, metric_fun in metric_fun.items():
            eval_result[name] = metric_fun(y, y_predict)
        return eval_result

    def recommend_k_item(self, user, dataset_val, k=1, is_softmax=True):  # for kuaishou data
        df_video_env = dataset_val.df_item_val
        # item_index = df_video_env.index.to_numpy()
        video = np.arange(len(df_video_env))
        # df_video_env.index.to_numpy()

        x = np.concatenate((np.ones([len(df_video_env), 1]) * user,
                            np.expand_dims(video, axis=-1),
                            df_video_env.values), 1)

        y_pred = np.zeros(len(x))
        for i in range(len(x)):
            xi = x[i, :]
            yi = self.linucb_arms[i].calc_UCB(xi)
            y_pred[i] = yi

        index = y_pred.argmax()

        item_index = df_video_env.index.to_numpy()

        recommendation = item_index[index]

        xi = x[index, :]

        reward = float(self.linucb_arms[index].calc_reward(xi))

        return recommendation, reward


def linucb_trainer(model, env, epoch, df_x, df_y, dataset_val, logger, metric_fun):
    x_np = df_x.to_numpy()
    y_np = df_y.to_numpy()

    # x_np = x_np[:10000, :]

    for epo in range(epoch):
        for i in tqdm(range(len(x_np)), desc="Epoch: {}".format(epo)):

            x = x_np[i, :]
            arm = int(x[1])
            if arm not in env.lbe_video.classes_:
                continue

            arm_small = env.lbe_video.transform([arm])[0]

            reward = y_np[i]

            model.linucb_arms[arm_small].reward_update(reward, x)

        eval_result = model.evaluate_data(dataset_val, metric_fun, env.lbe_video)
        eval_result_RL = test_static_model_in_RL_env(model, env=env, dataset_val=dataset_val, is_softmax=False)

        eval_result = {"val_" + k: v for k, v in eval_result.items()}
        eval_result_RL = {"RL_val_" + k: v for k, v in eval_result_RL.items()}
        result = {}
        result.update(eval_result)
        result.update(eval_result_RL)

        logger.info("Epoch: [{}], Info: [{}]".format(epo, result))
