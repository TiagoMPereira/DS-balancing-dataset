import pandas as pd
from sklearn.decomposition import PCA


def get_explained_variance_ratio(pca: PCA) -> float:
    return pca.explained_variance_ratio_.sum()

def fit_pca(dataframe: pd.DataFrame, n_components: int) -> PCA:
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(dataframe)
    return pca

def get_n_components(dataframe: pd.DataFrame, threshold: float) -> int:
    for i in range(len(dataframe.columns)):
        pca = fit_pca(dataframe, i)

        evr = pca.explained_variance_ratio_.sum()

        if evr >= threshold:
            return i

    return len(dataframe.columns)-1


if __name__ == "__main__":

    data = pd.read_csv("./datasets/synthetic_dataset2.csv")
    generated = pd.read_csv("./results/synthetic_dataset2-u_random_0-o_adasyn_0.5.csv")
    target = "class"
    x_col = [col for col in data.columns if col != target]

    n_components = get_n_components(data[x_col], 0.95)
    print(f"# components: {n_components}")

    original_pca = fit_pca(data[x_col], n_components)
    generated_pca = fit_pca(generated[x_col], n_components)

    print(f"EVR Original: {get_explained_variance_ratio(original_pca)}")
    print(f"EVR Generated: {get_explained_variance_ratio(generated_pca)}")
