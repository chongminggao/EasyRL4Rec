# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 7:56 下午
# @Author  : Chongming GAO
# @FileName: utils.py


import logzero
import torch
from logzero import logger
import os
# from tensorflow.python.keras.callbacks import Callback

# from util.upload import my_upload
import re

# my nas docker path
REMOTE_ROOT = "/root/Rethink_RL4RS"
class LoggerCallback_Update():
    def __init__(self, logger_path):
        super().__init__()
        self.LOCAL_PATH = logger_path
        self.REMOTE_ROOT = REMOTE_ROOT
        self.REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(logger_path))

    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, logs))

        # 2. upload logger
        # self.upload_logger()

    def upload_logger(self):
        try:
            my_upload(self.LOCAL_PATH, self.REMOTE_PATH, self.REMOTE_ROOT)
        except Exception:
            print("Failed: Uploading file [{}] to remote path [{}]".format(self.LOCAL_PATH, self.REMOTE_PATH))




class LoggerCallback_RL(LoggerCallback_Update):
    def __init__(self, logger_path):
        super().__init__(logger_path)

    def on_epoch_end(self, epoch, logs=None):
        num_test = logs["n/ep"]
        len_tra = logs["n/st"] / num_test
        R_tra = logs["rew"]
        ctr = R_tra / len_tra

        result = dict()
        result['num_test'] = num_test
        result['len_tra'] = len_tra
        result['R_tra'] = R_tra
        result['ctr'] = f"{ctr:.5f}"

        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, result))

        # 2. upload logger
        # self.upload_logger()


class LoggerCallback_Policy():
    def __init__(self, logger_path, force_length):
        self.LOCAL_PATH = logger_path
        self.REMOTE_ROOT = REMOTE_ROOT
        self.REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(logger_path))
        self.force_length = force_length

    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):
        def find_item_domination_results(prefix):
            pattern = re.compile(prefix + "ifeat_")
            res_domination = {}
            for k,v in results.items():
                res_search = re.match(pattern, k)
                # print(k, res_search)
                if res_search:
                    res_domination[k] = v
            return res_domination

        def get_one_result(prefix):
            num_test = results["n/ep"]
            len_tra = results[prefix + "n/st"] / num_test
            R_tra = results[prefix + "rew"]
            ctr = R_tra / len_tra

            res = dict()
            res['num_test'] = num_test
            res[prefix + 'CV'] = f"{results[prefix + 'CV']:.5f}"
            res[prefix + 'CV_turn'] = f"{results[prefix + 'CV_turn']:.5f}"
            res[prefix + 'ctr'] = f"{ctr:.5f}"
            res[prefix + 'len_tra'] = len_tra
            res[prefix + 'R_tra'] = R_tra

            if prefix + "all_feats" in results:
                res[prefix + 'all_feats'] = results[prefix + "all_feats"].tolist()

            return res

        results_all = {}
        for prefix in ["", "NX_0_", f"NX_{self.force_length}_"]:
            res = get_one_result(prefix)
            res_domination = find_item_domination_results(prefix)
            results_all.update(res)
            results_all.update(res_domination)

        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, results_all))

        # 2. upload logger
        # self.upload_logger()


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