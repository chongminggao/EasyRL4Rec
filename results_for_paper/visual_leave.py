# -*- coding: utf-8 -*-
# @Time    : 2023/1/18 15:40
# @Author  : Chongming GAO
# @FileName: visual_leave.py


import os

import re


import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from visual_utils import walk_paths, loaddata, organize_df

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def group_data(df_all, epoch=200, group_feat="R_tra"):
    data_r = df_all.loc[:epoch, group_feat]
    meandata = data_r.mean()

    pattern_name = re.compile("(.*)[-\s]leave[=]?(\d+)")

    df = pd.DataFrame()
    for k, v in meandata.items():
        # print(k)
        res = re.search(pattern_name, k)
        method = res.group(1)
        leave = res.group(2)
        df.loc[int(leave), method] = v

    df = df[sorted(df.columns)]
    df.sort_index(inplace=True)

    return df



def visual_leave_threshold(df_ks_grouped, df_kr_grouped, save_fig_dir, savename):
    all_method = sorted(set(df_ks_grouped.columns.to_list() + df_kr_grouped.columns.to_list()))
    df_ks_grouped.rename(columns={"epsilon-greedy": r'$\epsilon$-greedy'}, inplace=True)
    df_kr_grouped.rename(columns={"epsilon-greedy": r'$\epsilon$-greedy'}, inplace=True)

    df_ks_grouped.rename(columns={"Ours": 'DORL'}, inplace=True)
    df_kr_grouped.rename(columns={"Ours": 'DORL'}, inplace=True)



    all_method = ['DORL', 'MOPO', 'MBPO', 'IPS', 'BCQ', 'CQL', 'CRR', 'SQN', r'$\epsilon$-greedy', "UCB"]
    color = sns.color_palette(n_colors=len(all_method))
    markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<"]

    color_kv = dict(zip(all_method, color))
    marker_kv = dict(zip(all_method, markers))

    color1 = [color_kv[k] for k in df_ks_grouped.columns]
    color2 = [color_kv[k] for k in df_kr_grouped.columns]
    marker1 = [marker_kv[k] for k in df_ks_grouped.columns]
    marker2 = [marker_kv[k] for k in df_kr_grouped.columns]

    fig = plt.figure(figsize=(5.5, 2))
    plt.subplots_adjust(wspace=0.35)

    ax1 = plt.subplot(121)
    # ax1 = plt.gca()
    ax1.set_ylabel("Cumulative reward", fontsize=11)
    ax1.set_xlabel(r"Window size $N$", fontsize=11)
    df_ks_grouped.plot(kind="line", linewidth=1.8, ax=ax1, legend=None, color=color1, fillstyle='none', alpha=0.7, markeredgewidth=1.8)
    for i, line in enumerate(ax1.get_lines()):
        line.set_marker(marker1[i])
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid(linestyle='dashdot', linewidth=0.8)
    plt.xticks(range(1, 11), range(1, 11))
    ax1.set_title("KuaiRec", fontsize=11, y=1)

    ax2 = plt.subplot(122)
    ax2.set_ylabel("Cumulative reward", fontsize=11)
    ax2.set_xlabel(r"Window size $N$", fontsize=11)
    df_kr_grouped.plot(kind="line", linewidth=1.8, ax=ax2, legend=None, color=color2, fillstyle='none', alpha=0.7,
                       markeredgewidth=1.8)
    for i, line in enumerate(ax2.get_lines()):
        line.set_marker(marker2[i])
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid(linestyle='dashdot', linewidth=0.8)
    # plt.xticks(range(0, 5), ["1.0", "2.0", "3.0", "4.0", "5.0"])
    plt.xticks(range(1, 11), range(1, 11))
    ax2.set_title("KuaiRand", fontsize=11, y=1)


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    dict_label = dict(zip(labels1,lines1))
    dict_label.update(dict(zip(labels2,lines2)))
    # dict_label1 = {r'$\epsilon$-greedy' if k == 'Epsilon-greedy' else k: v for k, v in dict_label.items()}
    dict_label = {method: dict_label[method] for method in all_method}

    ax1.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=5,
               loc='lower left', columnspacing=1,
               bbox_to_anchor=(-0.15, 1.13), fontsize=9.5)


    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)

def visual_leave_condition():
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    print("Loading logs...")
    dir1 = os.path.join(".", "results_leave", "kuairec")
    filenames1 = walk_paths(dir1)
    df1 = loaddata(dir1, filenames1)

    dir2 = os.path.join(".", "results_leave", "kuairand")
    filenames2 = walk_paths(dir2)
    df2 = loaddata(dir2, filenames2)



    print("Transform data...")
    # ways={'FB'}
    ways = {"FB", "NX_0_", "NX_10_"}
    metrics={'ctr', 'len_tra', 'R_tra',  'CV', 'CV_turn', 'ifeat_feat'}


    df_kuaishou = organize_df(df1, ways, metrics)
    df_kuairand = organize_df(df2, ways, metrics)

    df_ks = df_kuaishou['NX_0_']
    df_kr = df_kuairand['NX_0_']

    df_ks_grouped = group_data(df_ks, group_feat="R_tra")
    df_kr_grouped = group_data(df_kr, group_feat="R_tra")

    print("Producing the figure...")
    visual_leave_threshold(df_ks_grouped, df_kr_grouped, save_fig_dir, savename="leave")
    print("Figure rendered.")


if __name__ == '__main__':
    # args = get_args()
    visual_leave_condition()
