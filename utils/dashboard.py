import pandas as pd
import numpy as np
from scale_funcs import *

DATASET_CODES_BINARY = ["37", "44", "1462", "1479", "1510"]
DATASET_CODES_MULTICLASS = ["23", "181", "1466", "40691", "40975"]
DATASET_CODES_MULTILABEL = ["41465", "41468", "41470", "41471", "41473"]

DATASET_DIAGNOSIS = {
    "23": "multi",
    "37": "bin",
    "44": "bin",
    "181": "multi",
    "1462": "bin",
    "1466": "multi",
    "1479": "bin",
    "1510": "bin",
    "40691": "multi",
    "40975": "multi",
    "41465": "multilabel",
    "41468": "multilabel",
    "41470": "multilabel",
    "41471": "multilabel",
    "41473": "multilabel"
}

DATASET_SCORE_ARAGAO = {
    "23": 0.615,
    "37": 0.804,
    "44": 0.966,
    "181": 0.632,
    "1462": 1.0,
    "1466": 1.0,
    "1479": 0.926,
    "1510": 1.0,
    "40691": 0.712,
    "40975": 1.0,
    "41465": 0.379,
    "41468": 0.516,
    "41470": 0.738,
    "41471": 0.799,
    "41473": 0.229
}

DATASET_MEAN_TIME_ARAGAO = {
    "23": 17,
    "37": 22,
    "44": 80,
    "181": 34,
    "1462": 11,
    "1466": 26,
    "1479": 17,
    "1510": 6,
    "40691": 219,
    "40975": 26,
    "41465": 50,
    "41468": 82,
    "41470": 141,
    "41471": 218,
    "41473": 393
}

def import_df(
    df_codes: list,
    path: str = "C:/Users/marce/git/DS-balancing-metric/artifacts/optuna_dbs_final/optuna_openml_"
):

    results = {}

    for dataset_code in df_codes:

        file_path = path+dataset_code+".csv"

        df = pd.read_csv(file_path, index_col=0)
        results[dataset_code] = df

    return results

def get_dataset_options():
    return ["Todos", "Binários", "Multiclasse", "Multilabel"] + DATASET_CODES_BINARY

def crop_df(dfs: dict, selected_value: str) -> dict:
    if selected_value == "Todos":
        return dfs
    if selected_value == "Binários":
        return {k: v for k, v in dfs.items() if DATASET_DIAGNOSIS[k] == "bin"}
    if selected_value == "Multiclasse":
        return {k: v for k, v in dfs.items() if DATASET_DIAGNOSIS[k] == "multi"}
    if selected_value == "Multilabel":
        return {k: v for k, v in dfs.items() if DATASET_DIAGNOSIS[k] == "multilabel"}
    return {selected_value: dfs[selected_value]}

def get_statistical_results(dfs: dict):
    statistic_results = {}
    for k in dfs.keys():
        df = dfs[k]
        df_total = len(df)
        valid_df = df.loc[df["value"] != -1]
        valid_df = valid_df.loc[~valid_df["value"].isna()]
        valid_df = valid_df.loc[valid_df["state"] == "COMPLETE"]
        df_completed = len(valid_df)

        # Scores
        min_score = min(valid_df["value"])
        max_score = max(valid_df["value"])
        mean_score = valid_df["value"].mean()
        std_score = valid_df["value"].std()

        # Time
        min_time = min(valid_df["total_time"])
        max_time = max(valid_df["total_time"])
        mean_time = valid_df["total_time"].mean()
        std_time = valid_df["total_time"].std()

        improvement_mean_time = mean_time - DATASET_MEAN_TIME_ARAGAO[k]

        statistic_results[k] = {
            "trials_total": df_total,
            "trials_completed": df_completed,
            "score_min": min_score,
            "score_max": max_score,
            "score_mean": mean_score,
            "score_stdev": std_score,
            "time_min": min_time,
            "time_max": max_time,
            "time_mean": mean_time,
            "time_stdev": std_time,
            "improved_time (mean)": improvement_mean_time
        }

    return pd.DataFrame(statistic_results).T

def get_optimized_results(dfs: dict):
    optimized_results = {}

    for k in dfs.keys():
        df = dfs[k]
        valid_df = df.loc[df["value"] != -1]
        valid_df = valid_df.loc[~valid_df["value"].isna()]
        valid_df = valid_df.loc[valid_df["state"] == "COMPLETE"]

        aragao = DATASET_SCORE_ARAGAO[k]
        optimized = max(valid_df["value"])

        # Combinations
        combination_df = valid_df.loc[(valid_df["oversampling_threshold"] != "0") & (valid_df["undersampling_threshold"] != "0")]
        combination_optimized = combination_df.loc[combination_df["value"] > aragao]

        # Only over
        over_df = valid_df.loc[valid_df["undersampling_threshold"] == "0"]
        over_optimized = over_df.loc[over_df["value"] > aragao]

        # Only under
        under_df = valid_df.loc[valid_df["oversampling_threshold"] == "0"]
        under_optimized = under_df.loc[under_df["value"] > aragao]

        optimized_results[k] = {
            "Aragão et al": aragao,
            "Optimized": optimized,
            "Improvement": optimized - aragao,
            "Combination": len(combination_optimized),
            "Undersampler": len(over_optimized),
            "Oversampler": len(under_optimized)
        }

    return pd.DataFrame(optimized_results).T

def get_highest_results(dfs: dict):
    best_trials = {}

    for k in dfs.keys():
        df = dfs[k]
        valid_df = df.loc[df["value"] != -1]
        valid_df = valid_df.loc[~valid_df["value"].isna()]
        valid_df = valid_df.loc[valid_df["state"] == "COMPLETE"]

        optimized = max(valid_df["value"])

        # Trials with best score
        trials_max_score = valid_df.loc[valid_df["value"] == optimized]
        trials_max_score.sort_values(by="total_time", ascending=True, inplace=True)

        best_trial = trials_max_score.head(1)[["oversampling_method", "oversampling_threshold", "undersampling_method", "undersampling_threshold", "total_time"]].values.tolist()[0]

        best_over_method = best_trial[0]
        best_over_thresh = best_trial[1]
        best_under_method = best_trial[2]
        best_under_thresh = best_trial[3]
        best_time = best_trial[4]

        best_trials[k] = {
            "Optimized": optimized,
            "Trials Highest Score": len(trials_max_score),
            "Oversampling method": best_over_method,
            "Oversampling threshold": best_over_thresh,
            "Undersampling method": best_under_method,
            "Undersampling threshold": best_under_thresh,
            "Time (s)": best_time
        }

    return pd.DataFrame(best_trials).T

def get_highest_hp(dfs: dict):
    best_dataframes = []
    for k in dfs.keys():
        df = dfs[k]
        valid_df = df.loc[df["value"] != -1]
        valid_df = valid_df.loc[~valid_df["value"].isna()]
        valid_df = valid_df.loc[valid_df["state"] == "COMPLETE"]

        optimized = max(valid_df["value"])

        # Trials with best score
        trials_max_score = valid_df.loc[valid_df["value"] == optimized]
        best_dataframes.append(trials_max_score)

    best_dataframes = pd.concat(best_dataframes)
    best_dataframes.loc[best_dataframes["oversampling_threshold"] == "0", "oversampling_method"] = "None"
    best_dataframes.loc[best_dataframes["undersampling_threshold"] == "0", "undersampling_method"] = "None"

    best_dataframes = best_dataframes[["undersampling_method", "oversampling_method"]]
    best_dataframes["n"] = [1] * len(best_dataframes)
    tabela_contagem = best_dataframes.pivot_table(index='undersampling_method', columns='oversampling_method', values='n', aggfunc=np.sum, fill_value=0, margins=True)
    return tabela_contagem

def are_equal(x, y, eps=1e-6):
    return abs(x - y) < eps

def get_custo_beneficio(dfs: dict):
    mean_dataframes = []
    best_dataframes = []
    for k in dfs.keys():
        df = dfs[k]
        baseline = DATASET_SCORE_ARAGAO[k]
        base_time = DATASET_MEAN_TIME_ARAGAO[k]
        valid_df = df.loc[df["value"] != -1]
        valid_df = valid_df.loc[~valid_df["value"].isna()]
        valid_df = valid_df.loc[valid_df["state"] == "COMPLETE"]

        # Concerns 13A and 13B
        # copia caso queira todos os trials
        # mean_df = valid_df.copy(deep=True)
        # filtro caso queira apenas os trials melhorados
        mean_df = valid_df.loc[(valid_df["value"] > baseline) | (are_equal(valid_df["value"], baseline) & (valid_df["total_time"] < base_time))]
        mean_df.loc[:, "score_mean"] = mean_df["value"] - baseline
        mean_df.loc[:, "time_delta_mean"] = mean_df["total_time"] - base_time
        mean_df.loc[:, "df_id"] = [k] * len(mean_df)
        print('k', k, 'mean_df', mean_df)
        mean_dataframes.append(mean_df)

        # x == y se (abs(x-y) < eps)
        improved_df = valid_df.loc[(valid_df["value"] > baseline) | (are_equal(valid_df["value"], baseline) & (valid_df["total_time"] < base_time))]
        if len(improved_df) == 0:
            continue
        improved_df.loc[:, "score_improve"] = improved_df["value"] - baseline
        improved_df.loc[:, "time_delta"] = improved_df["total_time"] - base_time
        improved_df.loc[:, "df_id"] = [k] * len(improved_df)
        best_dataframes.append(improved_df)

    mean_dataframes = pd.concat(mean_dataframes, axis=0)
    mean_dataframes.loc[mean_dataframes["oversampling_threshold"] == "0", "oversampling_method"] = "None"
    mean_dataframes.loc[mean_dataframes["undersampling_threshold"] == "0", "undersampling_method"] = "None"

    best_dataframes = pd.concat(best_dataframes, axis=0)
    best_dataframes.loc[best_dataframes["oversampling_threshold"] == "0", "oversampling_method"] = "None"
    best_dataframes.loc[best_dataframes["undersampling_threshold"] == "0", "undersampling_method"] = "None"

    over_methods = sorted(list(best_dataframes["oversampling_method"].unique()))
    under_methods = sorted(list(best_dataframes["undersampling_method"].unique()))

    counts, mean_scores, mean_times = {}, {}, {}
    full_scores = {}
    for o in over_methods:
        over_scores = {}
        counts[o] = {}
        mean_scores[o] = {}
        mean_times[o] = {}
        for u in under_methods:

            # Concerns 13A and 13B
            print(o, u)
            aux_dataset_mean = mean_dataframes.loc[(mean_dataframes["oversampling_method"] == o) & (mean_dataframes["undersampling_method"] == u)]
            if len(aux_dataset_mean) > 0:
                mean_scores[o][u] = aux_dataset_mean["score_mean"].median()
                mean_times[o][u] = aux_dataset_mean["time_delta_mean"].median()

            aux_dataset_abs = best_dataframes.loc[(best_dataframes["oversampling_method"] == o) & (best_dataframes["undersampling_method"] == u)]
            if len(aux_dataset_abs) == 0:
                continue
            counts[o][u] = len(aux_dataset_abs)
            sum_score_improve = sum(aux_dataset_abs["score_improve"])
            sum_time_delta = sum(aux_dataset_abs["time_delta"])
            custo_beneficio = sum_score_improve / sum_time_delta
            pass

            over_scores[u] = custo_beneficio
        full_scores[o] = over_scores

    print(best_dataframes["df_id"].value_counts())
    return pd.DataFrame(full_scores), best_dataframes, pd.DataFrame(counts), pd.DataFrame(mean_scores), pd.DataFrame(mean_times)

def generate_excel_report(dfs: dict, suffix: str):

    dataset_options = get_dataset_options()
    base_path = "C:/Users/marce/git/DS-balancing-metric/reports/"

    name = "5_custo_beneficio"    
    with pd.ExcelWriter(f'{base_path}{name}_{suffix}.xlsx') as writer:
        df_nome = name
        try:
            custo_beneficio, df_aux_custo, df_counts, mean_scores, mean_times = get_custo_beneficio(dfs)
        except:
            custo_beneficio=pd.DataFrame()

        custo_beneficio.fillna('-').to_excel(writer, sheet_name=df_nome, index=True)
        custo_beneficio.fillna(np.nan, inplace=True)
        print(custo_beneficio)

        numeric_cols = custo_beneficio.select_dtypes(include=['number'])

        min_value = numeric_cols.min().min()
        max_value = numeric_cols.max().max()
        abs_min = abs(min_value)
        abs_max = abs(max_value)
        new_min = -1
        new_max = +1

        scale_neg_fn = lambda x : (x / (abs_min**2)) if x < 0 else x
        custo_beneficio_scale_neg = custo_beneficio.copy(deep=True)
        custo_beneficio_scale_neg[numeric_cols.columns] = \
            custo_beneficio_scale_neg[numeric_cols.columns].map(scale_neg_fn)
        custo_beneficio_scale_neg.fillna('-').to_excel(writer, sheet_name=df_nome+"_scale_neg", index=True)

        scale_pos_fn = lambda x : (1 / np.sqrt(x)) if x > 0 else x
        custo_beneficio_scale_pos = custo_beneficio_scale_neg.copy(deep=True)
        custo_beneficio_scale_pos[numeric_cols.columns] = \
            custo_beneficio_scale_pos[numeric_cols.columns].map(scale_pos_fn)
        custo_beneficio_scale_pos.fillna('-').to_excel(writer, sheet_name=df_nome+"_scale_pos", index=True)

        min_value = custo_beneficio_scale_pos.min().min()
        max_value = custo_beneficio_scale_pos.max().max()
        force_range_fn = lambda x : (((x - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min)
        custo_beneficio_minmax = custo_beneficio_scale_pos.copy(deep=True)
        custo_beneficio_minmax[numeric_cols.columns] = \
            custo_beneficio_minmax[numeric_cols.columns].map(force_range_fn)
        custo_beneficio_minmax.fillna('-').to_excel(writer, sheet_name=df_nome+"_minmax", index=True)
        
        # excel_custom_format = +###.##0,00000;-###.##0,00000

        values_cont = custo_beneficio_scale_pos.fillna(0).values.ravel()
        percentiles = np.linspace(0.0, 1.0, 101)
        offset = int(np.ceil(len(percentiles) / 2))
        bins = np.quantile(values_cont, percentiles)
        values_disc = np.digitize(custo_beneficio_scale_pos.fillna(0), bins) - offset
        custo_beneficio_disc = custo_beneficio_scale_pos.copy(deep=True)
        for i in range(len(custo_beneficio_disc.values)):
            for j in range(len(custo_beneficio_disc.values[i])):
                custo_beneficio_disc.iat[i, j] = \
                    np.nan if np.isnan(custo_beneficio_disc.iat[i, j]) else values_disc[i, j]
        custo_beneficio_disc.fillna('-').to_excel(writer, sheet_name=df_nome+"_disc", index=True)

        # excel_custom_format = +#0;-#0

        df_counts.fillna('-').to_excel(writer, sheet_name="trial_counts", index=True)
        mean_scores.fillna('-').to_excel(writer, sheet_name="mean_scores", index=True)
        mean_times.fillna('-').to_excel(writer, sheet_name="mean_times", index=True)

    df_aux_custo.to_csv(base_path+name+"_aux.csv")

def rank_us_os_combinations_from_excel(file_path):
    # Load the data from the specified sheet in the Excel file
    df = pd.read_excel(file_path, sheet_name='5_custo_beneficio_disc', index_col=0)

    # Initialize an empty list to store the data in the desired format
    ranked_combinations = []

    # Loop through the DataFrame to extract oversampling, undersampling, and score information
    for undersampling_method in df.index:
        for oversampling_method in df.columns:
            score = df.loc[undersampling_method, oversampling_method]
            # Ensure score is a number, ignore if it's not
            if pd.notnull(score) and isinstance(score, (int, float)):
                ranked_combinations.append({
                    'Oversampling': oversampling_method,
                    'Undersampling': undersampling_method,
                    'Score': score
                })

    # Convert the list to a DataFrame
    ranked_df = pd.DataFrame(ranked_combinations)

    # Sort the DataFrame by 'Score' from lowest to highest
    ranked_df = ranked_df.sort_values(by='Score').reset_index(drop=True)

    # Write the ranked DataFrame to a new sheet in the existing Excel file
    with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='replace') as writer:
        ranked_df.fillna('None').to_excel(writer, sheet_name='ranking', index=False)


if __name__ == "__main__":

    dfs_binary = import_df(DATASET_CODES_BINARY)
    generate_excel_report(dfs_binary, 'binary')
    xlsx_path_binary = 'reports/5_custo_beneficio_binary.xlsx'
    rank_us_os_combinations_from_excel(xlsx_path_binary)

    dfs_multiclass = import_df(DATASET_CODES_MULTICLASS)
    generate_excel_report(dfs_multiclass, 'multiclass')
    xlsx_path_multiclass = 'reports/5_custo_beneficio_multiclass.xlsx'
    rank_us_os_combinations_from_excel(xlsx_path_multiclass)

    dfs_multilabel = import_df(DATASET_CODES_MULTILABEL)
    generate_excel_report(dfs_multilabel, 'multilabel')
    xlsx_path_multilabel = 'reports/5_custo_beneficio_multilabel.xlsx'
    rank_us_os_combinations_from_excel(xlsx_path_multilabel)