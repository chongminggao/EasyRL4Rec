# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 5:01 下午
# @Author  : Chongming GAO
# @FileName: evaluation.py
import numpy as np
import torch


from src.core.evaluation.utils import get_feat_dominate_dict
from src.core.evaluation.metrics import get_diversity, get_novelty

class Evaluator_Feat():
    def __init__(self, test_collector_set, df_item_val, need_transform, item_feat_domination, lbe_item, top_rate, draw_bar=False):
        self.collector_dict = test_collector_set.collector_dict
        # self.num_items = test_collector_set.env.get_env_attr("mat")[0].shape[1]

        # self.env = env
        self.df_item_val = df_item_val
        self.need_transform = need_transform
        self.item_feat_domination = item_feat_domination
        self.lbe_item = lbe_item
        self.top_rate = top_rate
        self.draw_bar = draw_bar

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
                acts = buffer.obs_next[indices][:, 1]  # buffer.act[indices]
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


        results_all = {}
        for name, collector in self.collector_dict.items():
            buffer = collector.buffer
            indices = results[name + "_idxs"] if name != "FB" else results["idxs"]
            feat_dominate_dict = get_actions_feat(buffer, indices, draw_bar=self.draw_bar)
            feat_dominate_dict_k = {name + "_" + k: v for k, v in
                                    feat_dominate_dict.items()} if name != "FB" else feat_dominate_dict
            results_all.update(feat_dominate_dict_k)

        results.update(results_all)

        return results
    

class Evaluator_Coverage_Count():
    def __init__(self, test_collector_set, df_item_val, need_transform):
        self.collector_dict = test_collector_set.collector_dict
        self.num_items = test_collector_set.env.get_env_attr("mat")[0].shape[1]

        # self.df_item_val = df_item_val
        self.need_transform = need_transform

    def on_epoch_begin(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):

        def get_count_results_for_one_collector(buffer):
            live_ind = np.ones([results["n/ep"]], dtype=bool)
            inds = buffer.last_index
            all_acts = []
            res = {}
            while any(live_ind):
                acts = buffer.obs_next[inds][:, 1]  # acts = buffer[inds].act
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

        results.update(results_all)

        return results

class Evaluator_User_Experience():
    def __init__(self, test_collector_set, df_item_val, item_similarity, item_popularity, need_transform, lbe_item):
        self.collector_dict = test_collector_set.collector_dict
        self.num_items = test_collector_set.env.get_env_attr("mat")[0].shape[1]
        self.need_transform = need_transform
        self.lbe_item = lbe_item

        self.df_item_val = df_item_val
        # self.item_similarity = test_collector_set.env.get_env_attr("df_similarity_small")[0]
        # self.item_popularity = test_collector_set.env.get_env_attr("item_popularity")[0]
        self.item_similarity = item_similarity
        self.item_popularity = item_popularity
        
    def on_epoch_begin(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):

        def get_actions(buffer, indices):

            num_tests = len(indices)
            live_mat = np.zeros([0, num_tests], dtype=bool)
            act_mat = np.zeros([0, num_tests], dtype=bool)
            rew_mat = np.zeros([0, num_tests], dtype=bool)
            is_end = np.zeros([num_tests], dtype=bool)

            while not all(is_end):
                acts = buffer.obs_next[indices][:, 1]  # buffer.act[indices]
                done = buffer.done[indices]
                rews = buffer.rew[indices]

                act_mat = np.vstack([act_mat, acts])
                rew_mat = np.vstack([rew_mat, rews])
                live_mat = np.vstack([live_mat, ~is_end])

                is_end[done] = True
                indices = buffer.next(indices)

            act_mat = act_mat.T
            rew_mat = rew_mat.T
            live_mat = live_mat.T

            all_acts = []
            all_rews = []
            all_acts_origin = []
            if self.need_transform:
                for i, live_arr in enumerate(live_mat):
                    all_acts.append(act_mat[i][live_arr])
                    all_rews.append(rew_mat[i][live_arr])
                    all_acts_origin.append(self.lbe_item.inverse_transform(act_mat[i][live_arr]))
            else:
                for i, live_arr in enumerate(live_mat):
                    all_acts.append(act_mat[i][live_arr])
                    all_rews.append(rew_mat[i][live_arr])
                    all_acts_origin.append(act_mat[i][live_arr])
        
            # if self.need_transform:
            #     all_acts_origin = self.lbe_item.inverse_transform(all_acts)
            # else:
            #     all_acts_origin = all_acts

            return all_acts_origin, all_rews, buffer.obs[indices][:, 0]

        
        def get_results_for_one_collector(actions, rews, users):
            res = {}
            all_div = []
            all_nov = []
            # all_ser = []
            for item_list, rew_list, user in zip(actions, rews, users):
                all_div.append(get_diversity(item_list, self.item_similarity))
                all_nov.append(get_novelty(item_list, self.item_popularity))
                # all_ser.append(get_serendipity(item_list, rew_list, user))
            res["Diversity"] = np.array(all_div).mean()
            res["Novelty"] = np.array(all_nov).mean()
            # res["ser"] = np.array(all_ser).mean()
            return res

        results_all = {}
        for name, collector in self.collector_dict.items():
            buffer = collector.buffer
            indices = results[name + "_idxs"] if name != "FB" else results["idxs"]
            actions, rews, users = get_actions(buffer, indices)
            
            res = get_results_for_one_collector(actions, rews, users)
            res_k = {name + "_" + k: v for k, v in res.items()} if name != "FB" else res
            results_all.update(res_k)

        results.update(results_all)


def save_model_fn(epoch, policy, model_save_path, optim, state_tracker, is_save=False):
    if not is_save:
        return
    model_save_path = model_save_path[:-3] + "-e{}".format(epoch) + model_save_path[-3:]
    # torch.save(model.state_dict(), model_save_path)
    torch.save({
        'policy': policy.state_dict(),
        'optim_RL': optim[0].state_dict(),
        'optim_state': optim[1].state_dict(),
        'state_tracker': state_tracker.state_dict(),
    }, model_save_path)
