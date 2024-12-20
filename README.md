# Extending K-Means Clustering with Ptolemy’s Inequality - Bachelor Thesis

## Motivation

k-Means clustering is one of the most widely used clustering methods. The Lloyd algorithm serves as the de facto standard for optimizing cluster centroids in k-Means by minimizing intra-cluster dissimilarity. Despite its age, Lloyd’s algorithm remains relevant due to numerous adaptations over time. One such adaptation is Elkan’s algorithm, which improves efficiency by leveraging the triangle inequality to reduce the number of distance computations during clustering.

This work explores the integration of **Ptolemy’s inequality**, a generalization of the triangle inequality, into the k-Means framework. The goal is to further enhance computational efficiency while maintaining clustering quality. The results demonstrate the theoretical and practical potential of this approach.

## Repository Structure

- **`means.py`**: Contains the implementation of the standard k-Means algorithm, Elkan’s algorithm, and the novel extension using Ptolemy’s inequality.
- **`notebook.ipynb`**: A Jupyter Notebook demonstrating how to use the `means.py` module and showcasing key functionalities.
- **`clustering_performance_results_formatted.csv`**: Contains the results of experimental evaluations, including performance metrics and comparisons across different implementations.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/kmeans-Extension-Ptolemy.git
