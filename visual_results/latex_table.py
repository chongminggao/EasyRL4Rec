import os
import pandas as pd
from results_utils import create_dir, load_dfs, handle_table

def combile_tables(dfs, used_way, save_table_dir, datasets, methods,
                       metrics = [r"$\text{R}_\text{cumu}$", "Length", r"$\text{R}_\text{avg}$"], 
                       final_rate=0.25,
                       savename="table",
                       precision=4):
    indices = [datasets, metrics]
    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices))

    for (df, dataset) in zip(dfs, datasets):
        df_latex, df_excel, df_avg = handle_table(df, final_rate, methods, precision)
        df_all[dataset] = df_latex[used_way]

    methods_order = dict(zip(methods, list(range(len(methods)))))
    df_all.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)

    # save
    filepath_latex = os.path.join(save_table_dir, f"{savename}.tex")
    with open(filepath_latex, "w") as file:
        file.write(df_all.style.to_latex(column_format='lccc', multicol_align='c'))
    print(f"{savename} latex tex file produced!")


def main():
    final_rate = 0.25
    force_length = 10
    rename_ways = ['Free', 'No Overlapping', "No Overlapping with {} turns".format(force_length)]
    _used_way = rename_ways[0]

    realpath = os.path.dirname(__file__)
    save_table_dir = os.path.join(realpath, "tables")
    create_dirs = [save_table_dir]
    create_dir(create_dirs)

    # 1. Batch RL
    env_list = ["Coat", "ML-1m", "KuaiRec"]
    metrics = {'R_tra', 'len_tra', 'ctr'}
    latex_metrics = [r"$\text{R}_\text{cumu}$", "Length", r"$\text{R}_\text{avg}$"]

    dirpath = os.path.join(realpath, "result_logs/1-batchRL")
    load_filepath_list = [os.path.join(dirpath, envname) for envname in env_list]
    savename = "table_1-batchRL"
    rename_cols = {
        "DiscreteBCQ": "BCQ",
        "DiscreteCQL": "CQL",
        "DiscreteCRR": "CRR",
        "SQN": "SQN"
    }
    dfs = load_dfs(load_filepath_list, metrics = metrics, rename_cols=rename_cols)
    combile_tables(dfs, _used_way, save_table_dir, env_list, list(rename_cols.values()), latex_metrics, final_rate, savename, precision=2)

    # 2. model-free RL
    env_list = ["Coat", "ML-1m", "KuaiRec"]
    metrics = {'R_tra', 'len_tra', 'ctr'}
    latex_metrics = [r"$\text{R}_\text{cumu}$", "Length", r"$\text{R}_\text{avg}$"]

    dirpath = os.path.join(realpath, "result_logs/2-model-freeRL")
    load_filepath_list = [os.path.join(dirpath, envname) for envname in env_list]
    savename = "table_2-model-freeRL"
    rename_cols = {
        "DQN": "DQN",
        "C51": "C51",
        "DDPG": "DDPG",
        "TD3": "TD3",
        "PG": "PG",
        "A2C": "A2C",
        "DiscretePPO": "PPO",
        "ContinuousPG": "PG(C)",
        "ContinuousA2C": "A2C(C)",
        "ContinuousPPO": "PPO(C)",
        "DORL": "DORL",
        "Intrinsic": "Intrinsic",
    }
    dfs = load_dfs(load_filepath_list, metrics = metrics, rename_cols=rename_cols)
    combile_tables(dfs, _used_way, save_table_dir, env_list, list(rename_cols.values()), latex_metrics, final_rate, savename)

    # 3. Coverage, Diversity, Novelty
    env_list = ["KuaiRec"]
    metrics = {'CV', 'Diversity', 'Novelty'}
    latex_metrics = ["Coverage", "Diversity", "Novelty"]

    dirpath = os.path.join(realpath, "result_logs/2-model-freeRL")
    load_filepath_list = [os.path.join(dirpath, envname) for envname in env_list]
    savename = "table_3-Coverage"
    rename_cols = {
        "DQN": "DQN",
        "C51": "C51",
        "DDPG": "DDPG",
        "TD3": "TD3",
        "PG": "PG",
        "A2C": "A2C",
        "DiscretePPO": "PPO",
        "ContinuousPG": "PG(C)",
        "ContinuousA2C": "A2C(C)",
        "ContinuousPPO": "PPO(C)",
        "DORL": "DORL",
        "Intrinsic": "Intrinsic",
    }
    dfs = load_dfs(load_filepath_list, metrics = metrics, rename_cols=rename_cols)
    combile_tables(dfs, _used_way, save_table_dir, env_list, list(rename_cols.values()), latex_metrics, final_rate, savename)

    # 4. StateTracker
    env_list = ["KuaiRec"]
    metrics = {'R_tra', 'len_tra', 'ctr'}
    latex_metrics = [r"$\text{R}_\text{cumu}$", "Length", r"$\text{R}_\text{avg}$"]

    dirpath = os.path.join(realpath, "result_logs/4-statetracker")
    load_filepath_list = [os.path.join(dirpath, envname) for envname in env_list]
    savename = "table_4-statetracker"
    rename_cols = {
        "A2C_gru": "GRU",
        "A2C_caser": "Caser",
        "A2C_sasrec": "SASRec",
        "A2C_avg": "Average",
        "A2C_nextitnet": "NextItNet",
    }
    dfs = load_dfs(load_filepath_list, metrics = metrics, rename_cols=rename_cols)
    combile_tables(dfs, _used_way, save_table_dir, env_list, list(rename_cols.values()), latex_metrics, final_rate, savename)

    # 5. Construction Method
    env_list = ["KuaiRec"]
    metrics = {'R_tra', 'len_tra', 'ctr'}
    latex_metrics = [r"$\text{R}_\text{cumu}$", "Length", r"$\text{R}_\text{avg}$"]

    dirpath = os.path.join(realpath, "result_logs/5-construction")
    load_filepath_list = [os.path.join(dirpath, envname) for envname in env_list]
    savename = "table_5-construction"
    rename_cols = {
        "CRR": "Sequential",
        "CRR_convolution": "Convolution",
        "CRR_counterfactual": "Counterfactual",
    }
    dfs = load_dfs(load_filepath_list, metrics = metrics, rename_cols=rename_cols)
    combile_tables(dfs, _used_way, save_table_dir, env_list, list(rename_cols.values()), latex_metrics, final_rate, savename)


if __name__ == '__main__':
    main()