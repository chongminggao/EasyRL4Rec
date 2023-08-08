# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 5:01 下午
# @Author  : Chongming GAO
# @FileName: evaluation.py
import numpy as np
import torch


from core.evaluation.utils import get_feat_dominate_dict

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

        results.update(results_all)

        return results

class Callback_User_Experience():
    def __init__(self, test_collector_set, df_item_val, need_transform, lbe_item):
        self.collector_dict = test_collector_set.collector_dict
        self.num_items = test_collector_set.env.get_env_attr("mat")[0].shape[1]
        self.need_transform = need_transform
        self.lbe_item = lbe_item

        # self.env = env
        self.df_item_val = df_item_val  ## val_items 3327

        # self.df_dist_small = test_collector_set.env.get_env_attr("df_dist_small")[0]
        # self.df_similarity_small = 1.0 / self.df_dist_small

        self.df_similarity_small = test_collector_set.env.get_env_attr("df_similarity_small")[0]

        # self.list_feat = test_collector_set.env.get_env_attr("list_feat")[0]  ## all_items 10728
        # self.list_users_history_category = test_collector_set.env.get_env_attr("list_users_history_category")[0]  ## all_users 7176
        self.item_popularity = test_collector_set.env.get_env_attr("item_popularity")[0]
        
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
                acts = buffer.act[indices]
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
    
        def cal_diversity(item_list):
            l = len(item_list)
            if l <= 1:
                return 1.0
            div = 0.0
            for i in range(l):
                for j in range(l):
                    if i < j:
                        div += (1-self.df_similarity_small.loc[item_list[i], item_list[j]])
            div /= (l * (l-1) / 2)
            return div
        
        def cal_serendipity(item_list, rew_list, user):
            hist_feat = eval(self.list_users_history_category[user])
            l = len(item_list)
            ser = 0.0
            for i in range(l):
                item_feat = self.list_feat[item_list[i]]
                # if rew_list[i] > 1.0 and not all(np.array(hist_feat)[item_feat]): ## TODO KuaiRec: reward_threshold serendipy_threshold
                if not all(np.array(hist_feat)[item_feat]):
                    ser += 1
            ser /= l
            return ser

        def cal_novelty(item_list):
            l = len(item_list)
            nov = 0.0
            for i in range(l):
                item_pop = self.item_popularity[item_list[i]]
                nov += (-np.log(item_pop))
            nov /= l
            return nov
        
        def get_results_for_one_collector(actions, rews, users):
            res = {}
            all_div = []
            # all_ser = []
            all_nov = []
            for item_list, rew_list, user in zip(actions, rews, users):
                all_div.append(cal_diversity(item_list))
                # all_ser.append(cal_serendipity(item_list, rew_list, user))
                all_nov.append(cal_novelty(item_list))
            res["div"] = np.array(all_div).mean()
            # res["ser"] = np.array(all_ser).mean()
            res["nov"] = np.array(all_nov).mean()
            return res

        results_all = {}
        for name, collector in self.collector_dict.items():
            buffer = collector.buffer
            indices = results[name + "_idxs"] if name != "FB" else results["idxs"]
            actions, rews, users = get_actions(buffer, indices)
            # assert False
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
