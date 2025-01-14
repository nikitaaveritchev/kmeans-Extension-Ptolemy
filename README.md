# Extending K-Means Clustering with Ptolemy’s Inequality - Bachelor Thesis

## Motivation

k-Means clustering is one of the most widely used clustering methods. The Lloyd algorithm serves as the de facto standard for optimizing cluster centroids in k-Means by minimizing intra-cluster dissimilarity. Despite its age, Lloyd’s algorithm remains relevant due to numerous adaptations over time. One such adaptation is Elkan’s algorithm, which improves efficiency by leveraging the triangle inequality to reduce the number of distance computations during clustering.

This work explores the integration of **Ptolemy’s inequality**, a generalization of the triangle inequality, into the k-Means framework. The goal is to further enhance computational efficiency while maintaining clustering quality. The results demonstrate the theoretical and practical potential of this approach.

## Repository Structure

- **`means.py`**: Contains the implementation of the standard k-Means algorithm, Elkan’s algorithm, and the novel extension using Ptolemy’s inequality.
- **`notebook.ipynb`**: A Jupyter Notebook demonstrating how to use the `means.py` module and showcasing key functionalities.
- **`clustering_performance_results_formatted.csv`**: Contains the results of experimental evaluations, including performance metrics and comparisons across different implementations.

## How to Use

0. Install prerequisites:
Building this paper requires python 3.9.21 or higher, lualatex 1.18.0, and GNU Make 4.4.1. It optionally requires qpdf 11.9.1 to linearize the resulting paper. See `pyproject.toml` for the version requirements of the python libraries used.

You might be able to use older version of the listed software, but we cannot guarantee compatibility or identical outputs. The code was run and paper created on Linux 6.12.8.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/kmeans-Extension-Ptolemy.git


## License

This project contains code under two licenses:

1. **CC BY 4.0**: All original contributions in this project are licensed under the Creative Commons Attribution 4.0 International License.
2. **MIT License**: Portions of this project are derived from [jjcordano's project](https://github.com/jjcordano/elkans_kmeans), which is licensed under the MIT license. These portions remain under the MIT license.

### Attribution
If using this project, please provide attribution for both:
- The original work by [jjcordano](https://github.com/jjcordano/elkans_kmeans) under the MIT license.
- This project under the CC BY 4.0 license.
