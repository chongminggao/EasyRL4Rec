# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np



CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "environments", "KuaiRand_Pure", "data")


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default='gaussian')
args = parser.parse_known_args()[0]


filename_train = os.path.join(DATAPATH, "log_standard_4_08_to_4_21_pure.csv")
df_train = pd.read_csv(filename_train)
filename_val = os.path.join(DATAPATH, "log_random_4_22_to_5_08_pure.csv")
df_val = pd.read_csv(filename_val)


if args.method == "maxmin":
    duration_max = max(max(df_train['duration_ms']),max(df_val['duration_ms']))
    duration_min = min(min(df_train['duration_ms']),min(df_val['duration_ms']))
    df_train['duration_normed'] = (df_train['duration_ms'] - duration_min) / (duration_max - duration_min)
    df_val['duration_normed'] = (df_val['duration_ms'] - duration_min) / (duration_max - duration_min)
else:
    df_duration = df_train[["video_id",'duration_ms']].append(df_val[["video_id",'duration_ms']])
    df_item_duration = df_duration.groupby("video_id").agg(np.mean)
    res = df_item_duration.describe()
    duration_mean = res.loc["mean"][0]
    duration_std = res.loc["std"][0]
    df_train['duration_normed'] = (df_train['duration_ms'] - duration_mean) / duration_std
    df_val['duration_normed'] = (df_val['duration_ms'] - duration_mean) / duration_std


df = df_train.append(df_val)

df['watch_ratio'] = df["play_time_ms"] / df["duration_ms"]
df.loc[np.isinf(df['watch_ratio']), 'watch_ratio'] = 1

df_train['watch_ratio'] = df_train["play_time_ms"] / df_train["duration_ms"]
df_train.loc[np.isinf(df_train['watch_ratio']), 'watch_ratio'] = 1

df_val['watch_ratio'] = df_val["play_time_ms"] / df_val["duration_ms"]
df_val.loc[np.isinf(df_val['watch_ratio']), 'watch_ratio'] = 1


if args.method == "maxmin":
    max_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(max)["watch_ratio"]
    min_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(min)["watch_ratio"]
    df_train["watch_ratio_normed"] = (df_train["watch_ratio"].to_numpy() - min_y.loc[df_train["video_id"]].to_numpy()) / \
                                   (max_y.loc[df_train["video_id"]].to_numpy() - min_y.loc[df_train["video_id"]].to_numpy())
    df_val["watch_ratio_normed"] = (df_val["watch_ratio"].to_numpy() - min_y.loc[df_val["video_id"]].to_numpy()) / \
                                   (max_y.loc[df_val["video_id"]].to_numpy() - min_y.loc[df_val["video_id"]].to_numpy())
else:
    mean_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(np.mean)["watch_ratio"]
    std_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(np.std)["watch_ratio"]
    df_train["watch_ratio_normed"] = (df_train["watch_ratio"].to_numpy() - mean_y.loc[df_train["video_id"]].to_numpy()) / \
                                   std_y.loc[df_train["video_id"]].to_numpy()
    df_val["watch_ratio_normed"] = (df_val["watch_ratio"].to_numpy() - mean_y.loc[df_val["video_id"]].to_numpy()) / \
                                     std_y.loc[df_val["video_id"]].to_numpy()
    df_train.loc[df_train["watch_ratio_normed"] > 10, "watch_ratio_normed"] = 10
    df_val.loc[df_val["watch_ratio_normed"] > 10, "watch_ratio_normed"] = 10
#
# mylist = df[["video_id", "watch_ratio"]].groupby("video_id").agg(list)["watch_ratio"]


df_train.loc[df_train["watch_ratio_normed"].isna(),"watch_ratio_normed"] = 0
df_val.loc[df_val["watch_ratio_normed"].isna(),"watch_ratio_normed"] = 0

df_train.rename(columns={"video_id":"item_id"}, inplace=True)
df_val.rename(columns={"video_id":"item_id"}, inplace=True)

df_train.to_csv(os.path.join(DATAPATH, "train_processed.csv"), index=False)
df_val.to_csv(os.path.join(DATAPATH, "test_processed.csv"), index=False)

print("saved_precessed_files!")