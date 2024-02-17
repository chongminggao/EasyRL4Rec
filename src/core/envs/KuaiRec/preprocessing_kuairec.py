# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np



# CODEPATH = os.path.dirname(__file__)
CODEPATH = "data/KuaiRec"
DATAPATH = os.path.join(CODEPATH, "data_raw")

def parse_args_processed():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--method", type=str, default='gaussian')
    parser.add_argument("--method", type=str, default='maxmin')
    args = parser.parse_known_args()[0]
    return args


filename_big = os.path.join(DATAPATH, "big_matrix.csv")
df_big = pd.read_csv(filename_big)
filename_small = os.path.join(DATAPATH, "small_matrix.csv")
df_small = pd.read_csv(filename_small)

filepath_processed_big = os.path.join(DATAPATH, "big_matrix_processed.csv")
filepath_processed_small = os.path.join(DATAPATH, "small_matrix_processed.csv")

def get_normalized_data():
    args = parse_args_processed()

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


    df = pd.concat([df_big, df_small], axis=0)

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
        max_thres = np.percentile(df_big["watch_ratio_normed"], 98)
        df_big.loc[df_big["watch_ratio_normed"] > max_thres, "watch_ratio_normed"] = max_thres
        df_small.loc[df_small["watch_ratio_normed"] > max_thres, "watch_ratio_normed"] = max_thres
    #
    # mylist = df[["video_id", "watch_ratio"]].groupby("video_id").agg(list)["watch_ratio"]

    df_big.loc[df_big["watch_ratio_normed"].isna(),"watch_ratio_normed"] = 0
    df_small.loc[df_small["watch_ratio_normed"].isna(),"watch_ratio_normed"] = 0

    df_big.rename(columns={"video_id":"item_id"}, inplace=True)
    df_small.rename(columns={"video_id":"item_id"}, inplace=True)

    df_big.to_csv(filepath_processed_big, index=False)
    df_small.to_csv(filepath_processed_small, index=False)

    print("saved_precessed_files!")
    return df_big, df_small


def get_df_data(filepath_input, usecols=None):
    filename = os.path.basename(filepath_input)
    assert filename == "big_matrix_processed.csv" or filename == "small_matrix_processed.csv"
    filepath = filepath_processed_big if filename == "big_matrix_processed.csv" else filepath_processed_small
    if os.path.exists(filepath):
        df_data = pd.read_csv(filepath, usecols=usecols)
    else:
        df_big, df_small = get_normalized_data()
        df_data = df_big if filename == "big_matrix_processed.csv" else df_small
        df_data = df_data[usecols] if usecols is not None else df_data
    return df_data

if __name__ == '__main__':
    df_big, df_small = get_normalized_data()