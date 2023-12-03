#!/bin/bash

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir torch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir tensorflow
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir scipy scikit-learn tqdm logzero pandas seaborn numba paramiko jupyterlab
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir gymnasium pettingzoo # for tianshou's new version


#pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade torch

#pip install -i https://mirrors.aliyun.com/pypi/simple/  --upgrade --no-cache-dir tensorflow

