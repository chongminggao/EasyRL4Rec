# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 5:01 下午
# @Author  : Chongming GAO
# @FileName: evaluation.py
import numpy as np
import torch
from tqdm import tqdm


def get_feat_dominate_dict(df_item_val, all_acts_origin, item_feat_domination, top_rate=0.8, draw_bar=False):
    if item_feat_domination is None:  # for yahoo
        return dict()
    # if need_transform:
    #     all_acts_origin = lbe_item.inverse_transform(all_acts)
    # else:
    #     all_acts_origin = all_acts

    feat_dominate_dict = {}
    recommended_item_features = df_item_val.loc[all_acts_origin]

    if "feat" in item_feat_domination:  # for kuairec and kuairand
        sorted_items = item_feat_domination["feat"]
        values = np.array([pair[1] for pair in sorted_items])
        values = values / sum(values)
        cumsum = values.cumsum()
        ind = 0
        for v in cumsum:
            if v > top_rate:
                break
            ind += 1
        if ind == 0:
            ind += 1
        dominated_values = np.array([pair[0] for pair in sorted_items])
        dominated_values = dominated_values[:ind]

        # dominated_value = sorted_items[0][0]
        recommended_item_features = recommended_item_features.filter(regex="^feat", axis=1)
        feat_numpy = recommended_item_features.to_numpy().astype(int)

        dominate_array = np.zeros([len(feat_numpy)], dtype=bool)
        for value in dominated_values:
            equal_mat = (feat_numpy == value)
            has_dominate = equal_mat.sum(axis=1)
            dominate_array = dominate_array | has_dominate

        rate = dominate_array.sum() / len(recommended_item_features)
        feat_dominate_dict["ifeat_feat"] = rate

        ####################################
        #Todo: visual bar plot
        if draw_bar:
            feat_predicted = feat_numpy
            cats_predicted = feat_predicted.reshape(-1)
            pos_cat_predicted = cats_predicted[cats_predicted > 0]

            feat_dominate_dict["all_feats"] = pos_cat_predicted

        ####################################

    else:  # for coat
        for feat_name, sorted_items in item_feat_domination.items():
            values = np.array([pair[1] for pair in sorted_items])

            values = values / sum(values)
            cumsum = values.cumsum()
            ind = 0
            for v in cumsum:
                if v > top_rate:
                    break
                ind += 1
            if ind == 0:
                ind += 1
            dominated_values = np.array([pair[0] for pair in sorted_items])
            dominated_values = dominated_values[:ind]

            # recommended_item_features = recommended_item_features.filter(regex="^feat", axis=1)
            feat_numpy = recommended_item_features[feat_name].to_numpy().astype(int)

            dominate_array = np.zeros([len(feat_numpy)], dtype=bool)
            for value in dominated_values:
                has_dominate = (feat_numpy == value)
                # has_dominate = equal_mat
                dominate_array = dominate_array | has_dominate

            rate = dominate_array.sum() / len(recommended_item_features)

            # dominated_value = sorted_items[0][0]
            # rate = (recommended_item_features[feat_name] == dominated_value).sum() / len(recommended_item_features)
            feat_dominate_dict["ifeat_" + feat_name] = rate

    return feat_dominate_dict


def interactive_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k, need_transform,
                           num_trajectory, item_feat_domination, remove_recommended, force_length=0, top_rate=0.8, draw_bar=False):
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0

    all_acts = []

    for i in tqdm(range(num_trajectory), desc=f"evaluate static method in {env.__str__()}"):
        user_ori = env.reset()
        if need_transform:
            user = env.lbe_user.inverse_transform(user_ori)[0]
        else:
            user = user_ori

        acts = []
        done = False
        while not done:
            recommended_id_transform, recommended_id_raw, reward_pred = model.recommend_k_item(
                user, dataset_val, k=k, is_softmax=is_softmax, epsilon=epsilon, is_ucb=is_ucb,
                recommended_ids=acts if remove_recommended else [])
            if need_transform:
                assert recommended_id_transform == env.lbe_item.transform([recommended_id_raw])[0]
            acts.append(recommended_id_transform)
            state, reward, done, info = env.step(recommended_id_transform)
            total_turns += 1
            # metric 1
            cumulative_reward += reward
            # metric 2
            click_loss = np.absolute(reward_pred - reward)
            total_click_loss += click_loss

            if done:
                if force_length > 0:  # do not end here
                    env.cur_user = user_ori
                    done = False
                else:
                    break
            if force_length > 0 and len(acts) >= force_length:
                done = True
                break

        all_acts.extend(acts)

    ctr = cumulative_reward / total_turns
    click_loss = total_click_loss / total_turns

    hit_item = len(set(all_acts))
    num_items = len(dataset_val.df_item_val)
    CV = hit_item / num_items
    CV_turn = hit_item / len(all_acts)

    # eval_result_RL = {"CTR": ctr, "click_loss": click_loss, "trajectory_len": total_turns / num_trajectory,
    #                   "trajectory_reward": cumulative_reward / num_trajectory}
    eval_result_RL = {
        "click_loss": click_loss,
        "CV": f"{CV:.5f}",
        "CV_turn": f"{CV_turn:.5f}",
        "ctr": ctr,
        "len_tra": total_turns / num_trajectory,
        "R_tra": cumulative_reward / num_trajectory}

    if need_transform:
        all_acts_origin = env.lbe_item.inverse_transform(all_acts)
    else:
        all_acts_origin = all_acts
    feat_dominate_dict = get_feat_dominate_dict(dataset_val.df_item_val, all_acts_origin, item_feat_domination, top_rate=top_rate)
    eval_result_RL.update(feat_dominate_dict)

    if remove_recommended:
        eval_result_RL = {f"NX_{force_length}_" + k: v for k, v in eval_result_RL.items()}

    return eval_result_RL


def test_static_model_in_RL_env(model, env, dataset_val, is_softmax=True, epsilon=0, is_ucb=False, k=1,
                                need_transform=False, num_trajectory=100, item_feat_domination=None, force_length=10, top_rate=0.8, draw_bar=False):
    eval_result_RL = {}

    eval_result_standard = interactive_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                                                  need_transform, num_trajectory, item_feat_domination,
                                                  remove_recommended=False, force_length=0, top_rate=top_rate, draw_bar=draw_bar)

    # No overlap and end with the env rule
    eval_result_NX_0 = interactive_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                                              need_transform, num_trajectory, item_feat_domination,
                                              remove_recommended=True, force_length=0, top_rate=top_rate, draw_bar=draw_bar)

    # No overlap and end with explicit length
    eval_result_NX_x = interactive_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                                              need_transform, num_trajectory, item_feat_domination,
                                              remove_recommended=True, force_length=force_length, top_rate=top_rate, draw_bar=draw_bar)

    eval_result_RL.update(eval_result_standard)
    eval_result_RL.update(eval_result_NX_0)
    eval_result_RL.update(eval_result_NX_x)

    return eval_result_RL


def test_taobao(model, env, epsilon=0):
    # test the model in the interactive system
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0
    num_trajectory = 100

    for i in range(num_trajectory):
        features = env.reset()
        done = False
        while not done:
            res = model(torch.FloatTensor(features).to(model.device).unsqueeze(0)).to('cpu').squeeze()
            item_feat_predict = res[model.y_index['feat_item'][0]:model.y_index['feat_item'][1]]
            action = item_feat_predict.detach().numpy()

            if epsilon > 0 and np.random.random() < epsilon:
                # Activate epsilon greedy
                action = np.random.random(action.shape)

            reward_pred = res[model.y_index['y'][0]:model.y_index['y'][1]]

            features, reward, done, info = env.step(action)

            total_turns += 1

            # metric 1
            cumulative_reward += reward

            # metric 2
            click_loss = np.absolute(float(reward_pred.detach().numpy()) - reward)
            total_click_loss += click_loss

            if done:
                break

    ctr = cumulative_reward / total_turns  # /10
    click_loss = total_click_loss / total_turns

    # print('CTR: %.2f'.format(ctr))
    eval_result_RL = {"CTR": ctr,
                      "click_loss": click_loss,
                      "trajectory_len": total_turns / num_trajectory,
                      "trajectory_reward": cumulative_reward / num_trajectory}  # /10}

    return eval_result_RL


class Callback_Coverage_Count():
    def __init__(self, test_collector_set, df_item_val, need_transform, item_feat_domination, lbe_item, top_rate, draw_bar=False):
        self.collector_dict = test_collector_set.collector_dict
        self.num_items = test_collector_set.env.get_env_attr("mat")[0].shape[1]

        # self.env = env
        self.df_item_val = df_item_val
        self.need_transform = need_transform
        self.item_feat_domination = item_feat_domination
        self.lbe_item = lbe_item
        self.top_rate = top_rate
        self.draw_bar=draw_bar

    def on_epoch_begin(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):

        def get_actions_feat(buffer, indices, draw_bar=False):

            num_tests = len(indices)
            live_mat = np.zeros([0, num_tests], dtype=bool)
            act_mat = np.zeros([0, num_tests], dtype=bool)

            is_end = np.zeros([num_tests], dtype=bool)

            # indices = results["idxs"]
            while not all(is_end):
                acts = buffer.act[indices]
                done = buffer.done[indices]

                act_mat = np.vstack([act_mat, acts])
                live_mat = np.vstack([live_mat, ~is_end])

                is_end[done] = True
                indices = buffer.next(indices)

            all_acts = act_mat[live_mat]

            if self.need_transform:
                all_acts_origin = self.lbe_item.inverse_transform(all_acts)
            else:
                all_acts_origin = all_acts
            feat_dominate_dict = get_feat_dominate_dict(self.df_item_val, all_acts_origin, self.item_feat_domination, top_rate=self.top_rate, draw_bar=draw_bar)

            return feat_dominate_dict

        def get_count_results_for_one_collector(buffer):
            live_ind = np.ones([results["n/ep"]], dtype=bool)
            inds = buffer.last_index
            all_acts = []
            res = {}
            while any(live_ind):
                acts = buffer[inds].act
                # print(acts)
                all_acts.extend(acts)

                live_ind = buffer.prev(inds) != inds
                inds = buffer.prev(inds[live_ind])

            hit_item = len(set(all_acts))
            res["CV"] = hit_item / self.num_items
            res["CV_turn"] = hit_item / len(all_acts)
            return res

        results_all = {}
        for name, collector in self.collector_dict.items():
            buffer = collector.buffer
            res = get_count_results_for_one_collector(buffer)
            res_k = {name + "_" + k: v for k, v in res.items()} if name != "FB" else res
            results_all.update(res_k)

            indices = results[name + "_idxs"] if name != "FB" else results["idxs"]
            feat_dominate_dict = get_actions_feat(buffer, indices, draw_bar=self.draw_bar)
            feat_dominate_dict_k = {name + "_" + k: v for k, v in
                                    feat_dominate_dict.items()} if name != "FB" else feat_dominate_dict
            results_all.update(feat_dominate_dict_k)

        results.update(results_all)

        return results

        # buffer = self.collector_dict.buffer
        # live_ind = np.ones([results["n/ep"]], dtype=bool)
        # inds = buffer.last_index
        # all_acts = []
        # while any(live_ind):
        #     acts = buffer[inds].act
        #     # print(acts)
        #     all_acts.extend(acts)
        #
        #     live_ind = buffer.prev(inds) != inds
        #     inds = buffer.prev(inds[live_ind])
        #
        # hit_item = len(set(all_acts))
        # results["CV"] = hit_item / self.num_items
        # results["CV_turn"] = hit_item / len(all_acts)
        # return results
