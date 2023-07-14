# -*- coding: utf-8 -*-
# @Time    : 2022/9/30 20:23
# @Author  : Chongming GAO
# @FileName: run_worldModel.py
import argparse
import os
import sys
import traceback

import logzero

from run_worldModel_ensemble import get_args_all, get_args_dataset_specific
from run_epsilongreedy import main

sys.path.extend(["./src", "./src/DeepCTR-Torch"])


def get_args_UCB():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_model_name", type=str, default="LinUCB")
    parser.add_argument('--is_ucb', dest='is_ucb', action='store_true')
    parser.add_argument('--no_ucb', dest='is_ucb', action='store_false')
    parser.set_defaults(is_ucb=True)
    parser.add_argument('--n_models', default=1, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument("--message", type=str, default="LinUCB")
    args = parser.parse_known_args()[0]
    return args



CODEPATH = os.path.dirname(__file__)




if __name__ == '__main__':
    args_all = get_args_all()
    args = get_args_dataset_specific(args_all.env)
    args_ucb = get_args_UCB()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_ucb.__dict__)

    try:
        main(args_all, is_save=False)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
