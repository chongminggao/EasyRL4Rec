# -*- coding: utf-8 -*-
# @Author  : Chongming GAO
# @FileName: visual_main_figure.py

import os
import re

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from visual_utils import walk_paths, organize_df, loaddata, create_dir, handle_table


def axis_shift(ax1, x_shift=0.01, y_shift=0):
    position = ax1.get_position().get_points()
    pos_new = position
    pos_new[:, 0] += x_shift
    pos_new[:, 1] += y_shift
    ax1.set_position(Bbox(pos_new))


def compute_improvement(df, col, last=0):
    our = df.iloc[-5:][col]["CIRS"].mean()
    prev = df.iloc[-last:][col]["CIRS w_o CI"].mean()
    print(f"Improvement on [{col}] of last [{last}] count is {(our - prev) / prev}")


def remove_redundent(df, level=1):
    methods = df.columns.levels[level]
    pattern_name = re.compile("(.*)[-\s]leave[=]?(\d+)")
    # methods = [pattern_name.match(method).group(1) for method in methods if pattern_name.match(method)]
    df.rename(columns={method:pattern_name.match(method).group(1) for method in methods if pattern_name.match(method)},
               level=level, inplace=True)
    df.rename(columns={"Ours": "DORL"}, level=level, inplace=True)


def load_dfs_to_visual(load_filepath_list, ways = {'FB', 'NX_0_', 'NX_10_'}, metrics = {'ctr', 'len_tra', 'R_tra'}):
    
    dfs = []
    for load_path in load_filepath_list:
        # result_dir1 = os.path.join(dirpath, envname)
        filenames = walk_paths(load_path)
        dfs1 = loaddata(load_path, filenames)
        df1 = organize_df(dfs1, ways, metrics)

        remove_redundent(df1, level=2)
        dfs.append(df1)
    
    return dfs
    

def visual_groups(dfs, save_fig_dir, group_names, savename, way="NX_0_"):
    visual_cols = ['R_tra', 'len_tra', 'ctr']

    series = "ABCDEFG"
    # group_names = ["KuaiRec", "KuaiRand"]
    # maxlen = [50, 100, 10, 30]
    fontsize = 11.5
    

    methods_list = set()
    for df in dfs:
        df = df[way]
        methods = df.columns.levels[1].to_list()
        methods_list.update(methods)
    methods_list = list(methods_list)

    num_methods = len(methods_list)

    colors = sns.color_palette(n_colors=num_methods)
    markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<", "x", "H"]
    if len(markers) < num_methods:
        markers = markers * (num_methods // len(markers) + 1)
                             
    markers = markers[:num_methods]
    
    color_kv = dict(zip(methods_list, colors))
    marker_kv = dict(zip(methods_list, markers))

    # methods_list = methods_list[::-1]
    # methods_order = dict(zip(methods_list, list(range(len(methods_list)))))

    fig = plt.figure(figsize=(5, 6))
    # plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    axs = []
    for index in range(len(dfs)):
        alpha = series[index]
        cnt = 1
        df = dfs[index][way]
        # methods_list = df.columns.levels[1].to_list()

        # df.sort_index(axis=1, key=lambda col: [methods_order[x] for x in col.to_list()], level=1, inplace=True)

        data_r = df[visual_cols[0]]
        data_len = df[visual_cols[1]]
        data_ctr = df[visual_cols[2]]



        color = [color_kv[name] for name in data_r.columns]
        marker = [marker_kv[name] for name in data_r.columns]

        ax1 = plt.subplot2grid((3, 2), (0, index))
        data_r.plot(kind="line", linewidth=1, ax=ax1, legend=None, color=color, markevery=int(len(data_r) / 10)+1,
                    fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax1.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, loc="left", x=0.4, y=.97)
        ax1.set_title("{}".format(group_names[index]), fontsize=fontsize, y=1.1, fontweight=400)
        cnt += 1

        ax2 = plt.subplot2grid((3, 2), (1, index))
        data_len.plot(kind="line", linewidth=1, ax=ax2, legend=None, color=color, markevery=int(len(data_r) / 10),
                      fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax2.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax2.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        # ax2.set_title(r"$\it{Max round=" + str(maxlen[index]) + r"}$", fontsize=fontsize - 1.5, loc="left", x=-0.2,
        #               y=.97)
        cnt += 1

        ax3 = plt.subplot2grid((3, 2), (2, index))
        data_ctr.plot(kind="line", linewidth=1, ax=ax3, legend=None, color=color, markevery=int(len(data_r) / 10),
                      fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax3.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        ax3.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        ax3.set_xlabel("epoch", fontsize=11)
        cnt += 1
        plt.grid(linestyle='dashdot', linewidth=0.8)
        if index == 2:
            axis_shift(ax1, .015)
            axis_shift(ax2, .015)
            axis_shift(ax3, .015)
        if index == 3:
            axis_shift(ax1, .005)
            axis_shift(ax2, .005)
            axis_shift(ax3, .005)
        axs.append((ax1, ax2, ax3))

    ax1, ax2, ax3 = axs[0]
    ax1.set_ylabel("Cumulative reward", fontsize=9, fontweight=400)
    ax2.set_ylabel("Interaction length", fontsize=9, fontweight=400)
    ax3.set_ylabel("Single-round reward", fontsize=9, fontweight=400)
    # ax3.yaxis.set_label_coords(-0.17, 0.5)

    dict_label = {}
    for group, df in enumerate(dfs):
        axx = axs[group][0]
        lines, labels = axx.get_legend_handles_labels()
        
        dict_label.update(dict(zip(labels, lines)))

    axx.legend(handles=dict_label.values(), labels=dict_label.keys(), fontsize=9.5, ncol=1, 
               loc='upper right', bbox_to_anchor=(2, 1))

    # # axo = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    # # x, y = np.array([[0.505, 0.505], [0.06, 0.92]])
    # # line = Line2D(x, y, lw=3, linestyle="dotted", color=(0.5, 0.5, 0.5))
    # # axo.add_line(line)
    # # plt.text(0.16, 0.02, "(A-B) Results with large interaction rounds", fontsize=11, fontweight=400)
    # # plt.text(0.58, 0.02, "(C-D) Results with limited interaction rounds", fontsize=11, fontweight=400)
    # # plt.axis('off')

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    # fig.savefig(os.path.join(save_fig_dir, savename + '.png'), format='png', bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("done!")


if __name__ == '__main__':
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)
    dirpath = os.path.join(realpath, "result_logs")

    visual_group_list = ["CoatEnv-v0"]  # ["EtsyEnv-v0"]
    load_filepath_list = [os.path.join(dirpath, envname) for envname in visual_group_list]

    metrics = {'ctr', 'len_tra', 'R_tra'}

    dfs = load_dfs_to_visual(load_filepath_list, metrics = metrics)

    way = "NX_0_"
    savename = "main_result"
    group_names = visual_group_list
    visual_groups(dfs, save_fig_dir, group_names, savename=savename, way=way)
