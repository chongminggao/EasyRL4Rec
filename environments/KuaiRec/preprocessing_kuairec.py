# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm



CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "data_raw")


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default='gaussian')
args = parser.parse_known_args()[0]


filename_big = os.path.join(DATAPATH, "big_matrix.csv")
df_big = pd.read_csv(filename_big)
filename_small = os.path.join(DATAPATH, "small_matrix.csv")
df_small = pd.read_csv(filename_small)



if args.method == "maxmin":
    duration_max = max(max(df_big['video_duration']),max(df_small['video_duration']))
    duration_min = min(min(df_big['video_duration']),min(df_small['video_duration']))
    df_big['duration_normed'] = (df_big['video_duration'] - duration_min) / (duration_max - duration_min)
    df_small['duration_normed'] = (df_small['video_duration'] - duration_min) / (duration_max - duration_min)
else:
    df_duration = df_big[["video_id",'video_duration']].append(df_small[["video_id",'video_duration']])
    df_item_duration = df_duration.groupby("video_id").agg(np.mean)
    res = df_item_duration.describe()
    duration_mean = res.loc["mean"][0]
    duration_std = res.loc["std"][0]
    df_big['duration_normed'] = (df_big['video_duration'] - duration_mean) / duration_std
    df_small['duration_normed'] = (df_small['video_duration'] - duration_mean) / duration_std


df = df_big.append(df_small)

if args.method == "maxmin":
    max_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(max)["watch_ratio"]
    min_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(min)["watch_ratio"]
    df_big["watch_ratio_normed"] = (df_big["watch_ratio"].to_numpy() - min_y.loc[df_big["video_id"]].to_numpy()) / \
                                   (max_y.loc[df_big["video_id"]].to_numpy() - min_y.loc[df_big["video_id"]].to_numpy())
    df_small["watch_ratio_normed"] = (df_small["watch_ratio"].to_numpy() - min_y.loc[df_small["video_id"]].to_numpy()) / \
                                   (max_y.loc[df_small["video_id"]].to_numpy() - min_y.loc[df_small["video_id"]].to_numpy())
else:

    mean_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(np.mean)["watch_ratio"]
    std_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(np.std)["watch_ratio"]
    df_big["watch_ratio_normed"] = (df_big["watch_ratio"].to_numpy() - mean_y.loc[df_big["video_id"]].to_numpy()) / \
                                   std_y.loc[df_big["video_id"]].to_numpy()
    df_small["watch_ratio_normed"] = (df_small["watch_ratio"].to_numpy() - mean_y.loc[df_small["video_id"]].to_numpy()) / \
                                     std_y.loc[df_small["video_id"]].to_numpy()
    df_big.loc[df_big["watch_ratio_normed"] > 10, "watch_ratio_normed"] = 10
    df_small.loc[df_small["watch_ratio_normed"] > 10, "watch_ratio_normed"] = 10
#
# mylist = df[["video_id", "watch_ratio"]].groupby("video_id").agg(list)["watch_ratio"]


df_big.loc[df_big["watch_ratio_normed"].isna(),"watch_ratio_normed"] = 0
df_small.loc[df_small["watch_ratio_normed"].isna(),"watch_ratio_normed"] = 0

df_big.rename(columns={"video_id":"item_id"}, inplace=True)
df_small.rename(columns={"video_id":"item_id"}, inplace=True)

df_big.to_csv(os.path.join(DATAPATH, "big_matrix_processed.csv"), index=False)
df_small.to_csv(os.path.join(DATAPATH, "small_matrix_processed.csv"), index=False)

print("saved_precessed_files!")