# -*- coding: utf-8 -*-
# @Time    : 2022/11/2 14:35
# @Author  : Chongming GAO
# @FileName: visual.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname="./SimHei.ttf")


def my_catplot(df, var="var", x="x", hue="domain", func=None, bin=100, is_sort=False, xrotation=0, yrotation=0, is_count=True):

    ax = sns.catplot(data=df, x="duration_s", hue="domain", kind="violin", bw=.25, cut=0, split=True)
    plt.savefig(f'cat_{var}_统计图.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'cat_{var}_统计图.png', bbox_inches='tight', pad_inches=0)

    plt.show()
    plt.close()


def visual_with_hue(df, var="var", x="x", hue="domain", func=None, bin=100, is_sort=False, xrotation=0, yrotation=0, is_count=True):
    if is_sort:
        def sort_fun(str):
            return float(str.strip("()").split(",")[0])

        df_ordered = pd.Categorical(df[x], sorted(df[x].unique(),key=sort_fun))
        df["temp"] = df_ordered
        ax = sns.countplot(data=df, x=x, hue="domain", order=sorted(df[x].unique(), key=sort_fun))
    else:
        ax = sns.histplot(data=df, x=x, hue=hue, bins=bin)

    if func:
        func(ax)

    gca = plt.gca()
    # fig_title = "Statistics of {}".format(df.name)
    # gca.set_title(fig_title, fontsize=14)
    gca.set_ylabel("数目", fontsize=14, fontproperties=myfont)
    gca.set_xlabel(var, fontsize=14, fontproperties=myfont)

    plt.xticks(rotation=xrotation)
    plt.yticks(rotation=yrotation)

    plt.savefig(f'连续_{var}_统计图.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'连续_{var}_统计图.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def visual_continue(df, var="var", func=None, bin=100, is_sort=False, xrotation=0, yrotation=0):
    if is_sort:
        def sort_fun(str):
            return float(str.strip("()").split(",")[0])

        df_ordered = pd.Categorical(df, sorted(df.unique(),key=sort_fun))
        ax = sns.histplot(data=df_ordered, bins=bin)
    else:
        ax = sns.histplot(data=df, bins=bin)

    if func:
        func(ax)

    gca = plt.gca()
    fig_title = "Statistics of {}".format(df.name)
    # gca.set_title(fig_title, fontsize=14)
    gca.set_ylabel("密度", fontsize=14, fontproperties=myfont)
    gca.set_xlabel(var, fontsize=14, fontproperties=myfont)

    plt.xticks(rotation=xrotation)
    plt.yticks(rotation=yrotation)

    plt.savefig(f'连续_{var}_统计图.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'连续_{var}_统计图.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def visual_statistics_discrete(df, var="my_variable", display_ratio=True, func=None, order=None, size=(6, 4.5), rotation=0, interval=0.02):
    ncount = len(df)

    fig = plt.figure(figsize=size)
    ax1 = fig.add_axes([0.14, 0.15, 0.74, 0.75])
    sns.countplot(x=df, color="#9fc5e8", linewidth=.6, edgecolor='k', ax=ax1, order=order)

    plt.grid(axis='y', linestyle='-.')

    gca = plt.gca()
    fig_title = "{}数目统计图".format(var)
    # gca.set_title(fig_title, fontsize=15, fontproperties=myfont)
    gca.set_ylabel("数目", fontsize=14, fontproperties=myfont)
    gca.set_xlabel(var, fontsize=14, fontproperties=myfont)

    ylim = ax1.get_ylim()
    plt.ylim(ylim[0], ylim[1] + interval * (ylim[1] - ylim[0]))

    if func:
        func(ax1)


    if display_ratio:
        # Make twin axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("比例 (%)", fontsize=14, fontproperties=myfont)

        for p in ax1.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax1.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                         ha='center', va='bottom', fontsize=10, rotation=rotation)  # set the alignment of the text

        ax2.set_ylim(0, ax1.get_ylim()[1] / ncount * 100)

    plt.savefig(f'离散_{var}_数目统计图.pdf', bbox_inches='tight',
                pad_inches=0)
    plt.savefig(f'离散_{var}_数目统计图.png', bbox_inches='tight',
                pad_inches=0)
    plt.show()
    plt.close(fig)
