import os
import json
from collections import OrderedDict
import pandas as pd
import re

def create_dir(create_dirs):
    """
    Create the required directories.
    """
    for dir in create_dirs:
        if not os.path.exists(dir):
            # logger.info('Create dir: %s' % dir)
            try:
                os.mkdir(dir)
            except FileExistsError:
                print("The dir [{}] already existed".format(dir))

def walk_paths(result_dir, sort=True):
    g = os.walk(result_dir)
    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name[0] == '.' or file_name[0] == '_':
                continue
            files.append(file_name)
    if sort:
        files.sort()
    return files

def loaddata(dirpath, filenames, use_filename=True):
    pattern_epoch = re.compile("Epoch: \[(\d+)]")
    pattern_info = re.compile("Info: \[(\{.+\})]")
    pattern_message = re.compile('"message": "(.+)"')
    pattern_array = re.compile("array\((.+?)\)")

    dfs = {}
    for filename in filenames:
        if filename[0] == '.' or filename[0] == '_':  # ".DS_Store":
            continue
        df = pd.DataFrame()
        message = "None"
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = False
            add = 0
            
            for i, line in enumerate(lines):
                res = re.search(pattern_epoch, line)
                if res:
                    epoch = int(res.group(1))
                    if (start == False) and epoch == 0:
                        add = 1
                        start = True
                    epoch += add
                    info = re.search(pattern_info, line)
                    try:
                        info1 = info.group(1).replace("\'", "\"")
                    except Exception as e:
                        print("jump incomplete line: [{}]".format(line))
                        continue
                    info2 = re.sub(pattern_array, lambda x: x.group(1), info1)

                    data = json.loads(info2)
                    df_data = pd.DataFrame(data, index=[epoch], dtype=float)
                    df = pd.concat([df, df_data])
                res_message = re.search(pattern_message, line)
                if res_message:
                    message = res_message.group(1)

            if use_filename:
                message = filename[:-4]

        dfs[message] = df

    dfs = OrderedDict(sorted(dfs.items(), key=lambda item: len(item[1]), reverse=True))
    return dfs

def load_dfs(load_filepath_list, 
             ways = {'FB', 'NX_0_', 'NX_10_'},
             metrics = {'R_tra', 'len_tra', 'ctr'},
             rename_cols=dict()):
    dfs = []
    for load_path in load_filepath_list:
        filenames = walk_paths(load_path)
        dfs1 = loaddata(load_path, filenames)
        df1 = organize_df(dfs1, ways, metrics, rename_cols=rename_cols)
        dfs.append(df1)
    return dfs

def organize_df(dfs, ways, metrics, rename_cols=None):
    indices = [list(dfs.keys()), ways, metrics]

    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Exp", "ways", "metrics"]))

    for message, df in dfs.items():
        for way in ways:
            for metric in metrics:
                col = (way if way != "FB" else "") + metric
                df_all[message, way, metric] = df[col]

    df_all.columns = df_all.columns.swaplevel(0, 2)
    df_all.sort_index(axis=1, level=0, inplace=True)
    df_all.columns = df_all.columns.swaplevel(0, 1)

    all_method = set(df_all.columns.levels[2].to_list())
    all_method_map = {}
    for method in all_method:
        res = re.match("\[([KT]_)?(.+?)(_len.+)?\]", method)
        if res:
            if res.group(2) in rename_cols.keys():
                all_method_map[method] = rename_cols[res.group(2)]
            else:
                all_method_map[method] = res.group(2)
            
    df_all.rename(
        columns=all_method_map,
        level=2, inplace=True)
    if rename_cols:
        df_all.drop(columns=[method for method in all_method_map.values() if method not in rename_cols.values()], 
                    level=2, inplace=True)
    return df_all

# utils for latex table
def get_top2_methods(col, is_largest):
    if is_largest:
        top2_name = col.nlargest(2).index.tolist()
    else:
        top2_name = col.nsmallest(2).index.tolist()
    name1, name2 = top2_name[0], top2_name[1]
    return name1, name2

def handle_one_col(df_metric, final_rate, is_largest, precision=4):
    length = len(df_metric)
    res_start = int((1 - final_rate) * length)
    mean = df_metric[res_start:].mean()
    # std = df_metric[res_start:].std()

    res_latex = pd.Series(map(lambda mean: f"${mean:.{precision}f}$", mean), # +{std:.4f}
                          index=mean.index)
    res_excel = pd.Series(map(lambda mean: f"{mean:.{precision}f}", mean),
                          index=mean.index)
    res_avg = mean

    name1, name2 = get_top2_methods(mean, is_largest=is_largest)
    res_latex.loc[name1] = r"$\mathbf{" + r"{}".format(res_latex.loc[name1][1:-1]) + r"}$"
    res_latex.loc[name2] = r"\underline{" + res_latex.loc[name2] + r"}"

    return res_latex, res_excel, res_avg

def handle_table(df_all, final_rate=1, methods=['BCQ'], precision=4):
    df_all.rename(columns={"FB": "Free", "NX_0_": r"No Overlapping", "NX_10_": r"No Overlapping with 10 turns"},
                  level=0, inplace=True)
    df_all.rename(columns={"R_tra": r"$\text{R}_\text{cumu}$", "ifeat_feat": "MCD",
                           "CV_turn": r"$\text{CV}_\text{M}$", "len_tra": "Length",
                           "ctr": r"$\text{R}_\text{avg}$",
                           "CV": "Coverage", "Diversity": "Diversity", "Novelty": "Novelty"}, level=1,
                  inplace=True)
    ways = df_all.columns.levels[0][::-1]
    metrics = df_all.columns.levels[1]
    if methods is None:
        methods = df_all.columns.levels[2].to_list()

    methods_order = dict(zip(methods, list(range(len(methods)))))

    df_latex = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    df_excel = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    df_avg = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))

    for col, way in enumerate(ways):
        df = df_all[way]
        for row, metric in enumerate(metrics):
            df_metric = df[metric]
            is_largest = False if metric == "MCD" else True
            df_latex[way, metric], df_excel[way, metric], df_avg[way, metric] = handle_one_col(df_metric, final_rate, is_largest, precision)

    df_latex.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)
    df_excel.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)
    df_avg.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)

    # print(df_latex.to_markdown())
    # excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    # df_excel.to_excel(excel_path)

    return df_latex, df_excel, df_avg