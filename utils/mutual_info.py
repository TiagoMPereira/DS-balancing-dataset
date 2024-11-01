from matplotlib import font_manager
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

def setup_fonts():
    font_dirs = ['./fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
        prop = font_manager.FontProperties(fname=font_file)
        print(font_file, prop.get_name())
    plt.rcParams['font.family'] = 'CMU Serif'

def get_stats(values: np.ndarray) -> dict:
    return {
        "size": len(values),
        "max": np.max(values),
        "min": np.min(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "Q1": np.percentile(values, 25),
        "Q3": np.percentile(values, 75),
        "Q3-Q1": np.percentile(values, 75) - np.percentile(values, 25),
        "stdev": np.std(values),
        "variance": np.var(values)
    }

datasets = [
    "openml_37", "openml_44", "openml_1462", "openml_1479", "openml_1510",
    "openml_23", "openml_181", "openml_1466",  "openml_40691", "openml_40975",
    "openml_41465", "openml_41468", "openml_41470", "openml_41471", "openml_41473"
]

setup_fonts()

mi_distributions = {}
mi_stats = {}

for i, ds in enumerate(datasets):
    print(ds)

    data = pd.read_csv(f"./datasets/{ds}.csv")
    ds_id = ds.split("_")[-1]

    X = data.drop(columns=["class"])
    y = data["class"]

    m_scores = mutual_info_classif(X, y)
    mi_distributions[f"{ds_id}"] = m_scores
    mi_stats[f"{ds_id}"] = get_stats(m_scores)

mi_stats = pd.DataFrame(mi_stats).T
mi_stats.to_csv("mutual_info.csv")

colors = [
    'red', 'red', 'red', 'red', 'red',
    'green', 'green', 'green', 'green', 'green',
    'blue', 'blue', 'blue', 'blue', 'blue'
]

plt.figure(figsize=(13,9))

boxplot = plt.boxplot(
    mi_distributions.values(), 
    patch_artist=True
)

# Loop over each box and set its face color
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

# Create custom legend handles
binary_patch = mpatches.Patch(color='red', label='Binary')
multiclass_patch = mpatches.Patch(color='green', label='Multiclass')
multilabel_patch = mpatches.Patch(color='blue', label='Multilabel')

# Add the legend
plt.legend(handles=[binary_patch, multiclass_patch, multilabel_patch], loc='upper right', fontsize=16)

plt.xticks(
    range(1, len(mi_distributions)+1),
    list(mi_distributions.keys()),
    fontsize=16
)
plt.yticks(fontsize=16)

plt.xlabel('Dataset IDs', fontsize=20)
plt.ylabel('Mutual Information Scores', fontsize=20)

plt.tight_layout()
plt.savefig('mutual_info.png')
plt.show()