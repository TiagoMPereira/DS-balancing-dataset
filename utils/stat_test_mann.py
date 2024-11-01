import json
from matplotlib import font_manager
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="FixedFormatter should only be used together with FixedLocator"
)

def setup_fonts():
    font_dirs = ['./fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
        prop = font_manager.FontProperties(fname=font_file)
        print(font_file, prop.get_name())
    plt.rcParams['font.family'] = 'CMU Serif'

def are_equal(x, y, eps=1e-6):
    return abs(x - y) < eps

def load_baseline_f1_scores(scenario, dataset_id):
    suffix = 'ps' if scenario == 'multilabel' else ''
    filename = f'./results_baseline/{dataset_id}{suffix}/automl_autogluon.json'
    with open(filename, 'r') as file:
        data = json.load(file)
    f1_scores = np.round([x['f1_score_weighted'] for x in data['results']], 3)
    best_f1 = np.max(f1_scores)
    mean_time = np.floor(np.mean([x['training_time'] for x in data['results']]))
    print(dataset_id, '[BASELINE ]', 'best_f1', best_f1, 'mean_time', mean_time)
    return f1_scores, best_f1, mean_time

def load_optimized_f1_scores(dataset_id, baseline_best_f1, baseline_mean_time):
    df = pd.read_csv(f'./results_optimized/optuna_openml_{dataset_id}.csv')
    df = df.loc[df['value'] != -1]
    df = df.loc[~df['value'].isna()]
    df = df.loc[df['state'] == 'COMPLETE']
    mean_time = df['total_time'].mean().round(2)
    df = df.loc[(df['oversampling_threshold'] != '0') | (df['undersampling_threshold'] != '0')]
    df = df.loc[(df['value'] > baseline_best_f1) | ((are_equal(df['value'], baseline_best_f1)) & (df['total_time'] < baseline_mean_time))]
    best_f1 = df['value'].max().round(3)
    print(dataset_id, '[OPTIMIZED]', len(df), 'trials were better than baseline')
    return df['value'].to_numpy(), best_f1, mean_time

setup_fonts()

datasets = {
    'binary': [37, 44, 1462, 1479, 1510],
    'multiclass': [23, 181, 1466, 40691, 40975],
    'multilabel': [41465, 41468, 41470, 41471, 41473],
}

results_dict = {}

alpha = 0.05  # significance level

for scenario, dataset_refs in datasets.items():

    plt.figure(figsize=(32, 18))
    plt.get_current_fig_manager().window.state('zoomed')
    # plt.subplots_adjust(hspace=0.4)
    plt_idx = 1
    results_dict[scenario] = {}

    for dataset_ref in dataset_refs:

        results_dict[scenario][dataset_ref] = {}
        
        try:

            baseline_scores, baseline_best_f1, baseline_mean_time = \
                load_baseline_f1_scores(scenario, dataset_ref)
            results_dict[scenario][dataset_ref]['baseline_samples'] = len(baseline_scores)
            results_dict[scenario][dataset_ref]['baseline_best_f1'] = baseline_best_f1
            results_dict[scenario][dataset_ref]['baseline_mean_time'] = baseline_mean_time

            optimized_scores, optimized_best_f1, optimized_mean_time = \
                load_optimized_f1_scores(dataset_ref, baseline_best_f1, baseline_mean_time)
            results_dict[scenario][dataset_ref]['optimized_samples'] = len(optimized_scores)
            results_dict[scenario][dataset_ref]['optimized_best_f1'] = optimized_best_f1
            results_dict[scenario][dataset_ref]['optimized_mean_time'] = optimized_mean_time

            try:
                shapiro_baseline = stats.shapiro(baseline_scores)
                print(dataset_ref, '[BASELINE ]', 'Shapiro-Wilk', 'p-value', shapiro_baseline.pvalue)
            except Exception as e:
                shapiro_baseline = None
                print(dataset_ref, '[BASELINE ]', 'Shapiro-Wilk', 'p-value', str(e))
            finally:
                results_dict[scenario][dataset_ref]['baseline_sw'] = \
                    shapiro_baseline.pvalue if shapiro_baseline else 'N/A'
                results_dict[scenario][dataset_ref]['baseline_is_normal'] = \
                    str(shapiro_baseline.pvalue >= alpha) if shapiro_baseline else 'N/A'
        
            try:
                shapiro_optimized = stats.shapiro(optimized_scores)
                print(dataset_ref, '[OPTIMIZED]', 'Shapiro-Wilk', 'p-value', shapiro_optimized.pvalue)
            except Exception as e:
                shapiro_optimized = None
                print(dataset_ref, '[OPTIMIZED]', 'Shapiro-Wilk', 'p-value', str(e))
            finally:
                results_dict[scenario][dataset_ref]['optimized_sw'] = \
                    shapiro_optimized.pvalue if shapiro_optimized else 'N/A'
                results_dict[scenario][dataset_ref]['optimized_is_normal'] = \
                    str(shapiro_optimized.pvalue >= alpha) if shapiro_optimized else 'N/A'

            plt.subplot(len(dataset_refs), 4, plt_idx)
            sns.histplot(baseline_scores, kde=True)
            plt.title(f'$F_1$ Score Histogram for Dataset {dataset_ref} (Baseline)', fontsize=14)
            plt.xlabel('$F_1$ Score', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.gca().set_xticklabels([f"{x:.3f}" for x in plt.gca().get_xticks()])
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt_idx += 1

            plt.subplot(len(dataset_refs), 4, plt_idx)
            stats.probplot(baseline_scores, dist="norm", plot=plt)
            plt.title(f'$F_1$ Score Q-Q Plot for Dataset {dataset_ref} (Baseline)', fontsize=14)
            plt.xlabel('Theoretical Quantiles', fontsize=12)
            plt.ylabel('$F_1$ Score', fontsize=12)
            plt.gca().set_xticklabels([f"{x:.2f}" for x in plt.gca().get_xticks()])
            plt.gca().set_yticklabels([f"{y:.3f}" for y in plt.gca().get_yticks()])
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt_idx += 1

            plt.subplot(len(dataset_refs), 4, plt_idx)
            sns.histplot(optimized_scores, kde=True)
            plt.title(f'$F_1$ Score Histogram for Dataset {dataset_ref} (Optimized)', fontsize=14)
            plt.xlabel('$F_1$ Score', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.gca().set_xticklabels([f"{x:.3f}" for x in plt.gca().get_xticks()])
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt_idx += 1

            plt.subplot(len(dataset_refs), 4, plt_idx)
            stats.probplot(optimized_scores, dist="norm", plot=plt)
            plt.title(f'$F_1$ Score Q-Q Plot for Dataset {dataset_ref} (Optimized)', fontsize=14)
            plt.xlabel('Theoretical Quantiles', fontsize=12)
            plt.ylabel('$F_1$ Score', fontsize=12)
            plt.gca().set_xticklabels([f"{x:.2f}" for x in plt.gca().get_xticks()])
            plt.gca().set_yticklabels([f"{y:.3f}" for y in plt.gca().get_yticks()])
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt_idx += 1

            u_stat, u_pvalue = mannwhitneyu(baseline_scores, optimized_scores, alternative='two-sided')
            print(dataset_ref, '[STATISTIC]', 'Mann-Whitney', 'U-statistic', u_stat, 'p-value', u_pvalue)
            
            if u_pvalue < alpha:
                print(dataset_ref, '[STATISTIC]', "The difference is statistically significant (p < 0.05) => not random chance")
            else:
                print(dataset_ref, '[STATISTIC]', "The difference is not statistically significant (p >= 0.05) => probably random chance")

            results_dict[scenario][dataset_ref]['mann_stat'] = u_stat
            results_dict[scenario][dataset_ref]['mann_p_value'] = u_pvalue
            results_dict[scenario][dataset_ref]['alpha'] = alpha
            results_dict[scenario][dataset_ref]['p_value < alpha'] = str(u_pvalue < alpha)
            results_dict[scenario][dataset_ref]['significant'] = str(u_pvalue < alpha)
    
        except FileNotFoundError as e:
            print('FileNotFoundError', e)
        print()

    plt.tight_layout()
    plt.savefig(f'figure-stat_test_{scenario}.png', dpi=300)
    # plt.show()

result_dfs = [pd.DataFrame.from_dict(v).T.assign(**{'scenario': k, 'id': v.keys()}) for k,v in results_dict.items()]
col_order = ['scenario', 'id', 
             'baseline_samples', 'baseline_best_f1', 'baseline_mean_time', 'baseline_sw', 'baseline_is_normal',
             'optimized_samples', 'optimized_best_f1', 'optimized_mean_time', 'optimized_sw', 'optimized_is_normal',
             'mann_stat', 'mann_p_value', 'alpha', 'p_value < alpha', 'significant']

summary_df = pd.concat(result_dfs)[col_order]

with pd.ExcelWriter('stat_test_mann.xlsx', engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, index=False, sheet_name='Summary')
    workbook = writer.book
    worksheet = writer.sheets['Summary']

    # Define specific formats for each column
    text_format = workbook.add_format({'num_format': '@'})                   # String format for 'scenario'
    integer_format = workbook.add_format({'num_format': '0'})                # Integer format for 'id' and sample counts
    float_3dec_format = workbook.add_format({'num_format': '0.000'})         # 3 decimal places for F1 scores
    float_2dec_format = workbook.add_format({'num_format': '0.00'})          # 2 decimal places for mean time and alpha
    scientific_format = workbook.add_format({'num_format': '0.00E+00'})      # Scientific notation for SW test p-values and stats

    # Apply formats to columns by name or index
    worksheet.set_column('A:A', None, text_format)        # 'scenario'
    worksheet.set_column('B:B', None, integer_format)     # 'id'
    worksheet.set_column('C:C', None, integer_format)     # 'baseline_samples'
    worksheet.set_column('D:D', None, float_3dec_format)  # 'baseline_best_f1'
    worksheet.set_column('E:E', None, float_2dec_format)  # 'baseline_mean_time'
    worksheet.set_column('F:F', None, scientific_format)  # 'baseline_sw'
    worksheet.set_column('G:G', None, text_format)        # 'baseline_is_normal'
    worksheet.set_column('H:H', None, integer_format)     # 'optimized_samples'
    worksheet.set_column('I:I', None, float_3dec_format)  # 'optimized_best_f1'
    worksheet.set_column('J:J', None, float_2dec_format)  # 'optimized_mean_time'
    worksheet.set_column('K:K', None, scientific_format)  # 'optimized_sw'
    worksheet.set_column('L:L', None, text_format)        # 'optimized_is_normal'
    worksheet.set_column('M:M', None, scientific_format)  # 'mann_stat'
    worksheet.set_column('N:N', None, scientific_format)  # 'mann_p_value'
    worksheet.set_column('O:O', None, float_2dec_format)  # 'alpha'
    worksheet.set_column('P:P', None, text_format)        # 'p_value < alpha'
    worksheet.set_column('Q:Q', None, text_format)        # 'significant'
