import numpy as np
import pandas as pd
from means import Kmeans, DataGenerator
from sklearn.datasets import make_moons, make_circles, load_iris, load_wine
import matplotlib.pyplot as plt
import seaborn as sns

import time
from tqdm import tqdm

def generate_datasets():
    datasets = {}

    # Gaussian datasets (Blobs)
    low_dim_generator = DataGenerator(n_samples=10000, n_features=10, n_clusters=4, random_state=43)
    low_dim_data, _ = low_dim_generator.generate_data()
    datasets["Gaussian_low"] = low_dim_data

    medium_dim_generator = DataGenerator(n_samples=10000, n_features=100, n_clusters=4, random_state=43)
    medium_dim_data, _ = medium_dim_generator.generate_data()
    datasets["Gaussian_medium"] = medium_dim_data

    high_dim_generator = DataGenerator(n_samples=10000, n_features=300, n_clusters=4, random_state=43)
    high_dim_data, _ = high_dim_generator.generate_data()
    datasets["Gaussian_high"] = high_dim_data

    # Iris dataset
    iris = load_iris()
    datasets['Iris'] = iris.data

    # Wine dataset
    wine = load_wine()
    datasets['Wine'] = wine.data

    # Moons dataset
    X_moons, _ = make_moons(n_samples=10000, noise=0.1, random_state=42)
    datasets['Moons'] = X_moons

    # Circles dataset
    X_circles, _ = make_circles(n_samples=15000, noise=0.1, factor=0.5, random_state=42)
    datasets['Circles'] = X_circles

    # Random datasets
    datasets["Random_low"] = np.random.uniform(0, 10, size=(5000, 10))
    datasets["Random_medium"] = np.random.uniform(0, 10, size=(5000, 100))
    datasets["Random_high"] = np.random.uniform(0, 10, size=(5000, 300))

    return datasets


def run_clustering(datasets, methods, k_values):
    results = {
        "Dataset": [],
        "Method": [],
        "K": [],
        "Distance_Evaluations": [],
        "Wall_Clock_Time": [],
    }
    progress_bar = tqdm(total=len(datasets) * len(k_values) * len(methods))
    for name, data in datasets.items():
        for k in k_values:
            for method in methods:
                progress_bar.set_postfix_str(f"{name}, k={k}, {method}")

                model = Kmeans(k=k, method=method)
                start_time = time.perf_counter()
                model.fit(data)
                end_time = time.perf_counter()
                results["Dataset"].append(name)
                results["Method"].append(method)
                results["K"].append(k)
                results["Distance_Evaluations"].append(model.distance_evaluations)
                results["Wall_Clock_Time"].append(end_time - start_time)

                progress_bar.update()
    return pd.DataFrame(results)


def save_results(
    results, filename="results/clustering_performance_results_formatted.xlsx"
):
    dim_suffixes = ("_low", "_medium", "_high")
    has_dims = results["Dataset"].str.endswith(dim_suffixes)
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        results[~has_dims].to_excel(writer, sheet_name="Results1", index=False)
        results[has_dims].to_excel(writer, sheet_name="Results2", index=False)


def plot_performance():
    file_path = 'clustering_performance_results_formatted.xlsx'
    data = pd.ExcelFile(file_path)
    df2 = data.parse('Results2')
    df1 = data.parse('Results1')

    df2[['Dataset', 'Dimension']] = df2['Dataset'].str.split('_', expand=True)
    df2['Dataset'] = df2['Dataset'].replace('Random', 'Uniform')

    df_filtered_gaussian = df2[(df2['K'].isin([3, 20, 100])) & (~df2['Method'].isin(['classic', 'Elkan']))]
    k_mapping = {3: 'K=3', 20: 'K=20', 100: 'K=100'}
    df_filtered_gaussian['K'] = df_filtered_gaussian['K'].map(k_mapping)

    palette = sns.color_palette("husl", len(df_filtered_gaussian['Method'].unique()))
    fig = plt.figure(figsize=(18, 15))
    datasets = df_filtered_gaussian['Dataset'].unique()
    dimensions = ['low', 'medium', 'high']

    for i, dataset in enumerate(datasets):
        for j, dimension in enumerate(dimensions):
            ax = plt.subplot2grid((3, 4), (i, j + 1))
            df_plot = df_filtered_gaussian[(df_filtered_gaussian['Dataset'] == dataset) & (df_filtered_gaussian['Dimension'] == dimension)]
            sns.lineplot(data=df_plot, x='K', y='Speedup', hue='Method', ax=ax, marker='o', palette=palette, legend=(i == 0 and j == 0))
            if i == 0:
                ax.set_title(f'{dimension.capitalize()} dimension')
            if j == 0:
                ax.set_ylabel('Speedup')
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')

        ax_annotate = plt.subplot2grid((3, 4), (i, 0))
        ax_annotate.annotate(dataset, xy=(0.5, 0.5), xytext=(-ax_annotate.yaxis.labelpad - 5, 0),
                             xycoords='axes fraction', textcoords='offset points', size=16, ha='center', va='center')
        ax_annotate.axis('off')

    df_filtered_iris_wine = df1[(df1['Dataset'].isin(['Iris', 'Wine'])) & (~df1['Method'].isin(['classic', 'Elkan']))]
    df_filtered_iris_wine['K'] = df_filtered_iris_wine['K'].map(k_mapping)

    datasets_iris_wine = df_filtered_iris_wine['Dataset'].unique()
    for j, dataset in enumerate(datasets_iris_wine):
        ax = plt.subplot2grid((3, 4), (2, j + 1))
        df_plot = df_filtered_iris_wine[df_filtered_iris_wine['Dataset'] == dataset]
        sns.lineplot(data=df_plot, x='K', y='Speedup', hue='Method', ax=ax, marker='o', palette=palette, legend=False)
        ax.set_title(f'{dataset} Dataset')
        ax.set_xlabel('K')
        ax.set_ylabel('Speedup')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig('combined_plot.pdf', format='pdf')
    plt.show()


def main():
    datasets = generate_datasets()
    methods = ['classic', 'Elkan', 'Ptolemy_upper', 'Ptolemy_lower', 'Ptolemy']
    k_values = [3, 20, 100]
    results = run_clustering(datasets, methods, k_values)
    save_results(results)
    plot_performance()


if __name__ == "__main__":
    main()