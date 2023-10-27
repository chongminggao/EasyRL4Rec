import os
import re
import pandas as pd

from visual_utils import walk_paths, organize_df, loaddata, create_dir, handle_table

def combile_two_tables(dfs, used_way, save_table_dir, methods, 
                       metrics = [r"$\text{R}_\text{cumu}$", r"$\text{R}_\text{avg}$", "Length"], 
                       final_rate=0.25,
                       savename="main_result"):
    datasets = ["Coat"]  # , "MovieLens"
    indices = [datasets, metrics]
    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices))

    for (df, dataset) in zip(dfs, datasets):
        # df = df[used_way]
        # methods = df.columns.levels[1].to_list()
        # methods_list.update(methods)

        df_latex, df_excel, df_avg = handle_table(df, final_rate=final_rate, methods=methods)
        # df_latex2, df_excel2, df_avg2 = handle_table(df2, final_rate=final_rate, methods=methods)

        df_all[dataset] = df_latex[used_way]

    methods_order = dict(zip(methods, list(range(len(methods)))))
    df_all.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)

    # save
    filepath_latex = os.path.join(save_table_dir, f"{savename}_table.tex")
    with open(filepath_latex, "w") as file:
        file.write(df_all.to_latex(escape=False, column_format='lccc', multicolumn_format='c'))
    print("latex tex file produced!")

def load_dfs(load_filepath_list, 
             ways = {'FB', 'NX_0_', 'NX_10_'},
             metrics = {'ctr', 'len_tra', 'R_tra'},
             rename_cols=None):
    
    dfs = []
    for load_path in load_filepath_list:
        # result_dir1 = os.path.join(dirpath, envname)
        filenames = walk_paths(load_path)
        dfs1 = loaddata(load_path, filenames)
        df1 = organize_df(dfs1, ways, metrics, rename_cols=rename_cols)

        # remove_redundent(df1, level=2)
        dfs.append(df1)
    
    return dfs

def main():
    realpath = os.path.dirname(__file__)
    final_rate = 0.25
    force_length = 10
    random_seed = 2023
    force_length_ways = 'NX_{}_'.format(force_length)
    ways = {'FB', 'NX_0_', force_length_ways}

    save_sub_dir = "table_NX_{}_{}".format(force_length, random_seed)
    dirpath = os.path.join(realpath, "result_logs")

    env_list = ["CoatEnv-v0"]
    load_filepath_list = [os.path.join(dirpath, envname) for envname in env_list]

    save_table_dir = os.path.join(realpath, "tables")
    create_dirs = [save_table_dir]
    create_dir(create_dirs)

    
    # metrics = {'ctr', 'len_tra', 'R_tra'}
    metrics = {'CV', 'Diversity', 'Novelty'}

    rename_cols = {
        "DiscreteBCQ": "BCQ",
        "DiscreteCQL": "CQL",
        "DQN": "DQN",
        "C51": "C51",
        "PG": "PG",
        "A2C": "A2C",
        "DiscretePPO": "PPO",
        "ContinuousPPO": "PPO(C)",
    }

    dfs = load_dfs(load_filepath_list, metrics = metrics, rename_cols=rename_cols)

    # _used_way = "No Overlapping with {} turns".format(force_length)
    _used_way = "No Overlapping"
    # _used_way = "Free"

    # latex_metrics = [r"$\text{R}_\text{cumu}$", r"$\text{R}_\text{avg}$", "Length"]
    latex_metrics = ["Cov", "Div", "Nov"]
    combile_two_tables(dfs, used_way=_used_way, save_table_dir=save_table_dir, methods=list(rename_cols.values()), metrics=latex_metrics, final_rate=final_rate, savename=save_sub_dir)


if __name__ == '__main__':
    main()