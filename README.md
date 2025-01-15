# Extending K-Means Clustering with Ptolemy’s Inequality

## Motivation

k-Means clustering is one of the most widely used clustering methods. The Lloyd algorithm serves as the de facto standard for optimizing cluster centroids in k-Means by minimizing intra-cluster dissimilarity. Despite its age, Lloyd’s algorithm remains relevant due to numerous adaptations over time. One such adaptation is Elkan’s algorithm, which improves efficiency by leveraging the triangle inequality to reduce the number of distance computations during clustering.

This work explores the integration of **Ptolemy’s inequality**, a property in Euclidean metric spaces similar to the triangle inequality, into the k-Means framework. The goal is to further enhance computational efficiency while maintaining clustering quality. The results demonstrate the theoretical and practical potential of this approach.

## Repository Structure

- **`main.py`**: Runs all experiments for the paper and creates the `results/*`; it calls:
- **`means.py`**: Contains the implementation of the standard k-Means algorithm, Elkan’s algorithm, and the novel extension using Ptolemy’s inequality.
- **`plot_results.py`**: Creates the singular plot of the paper.

The repo also contains:
- **`notebook.ipynb`**: A Jupyter Notebook demonstrating how to use the `means.py` module and showcasing key functionalities.
- **`Makefile`**: Programmatic overview of how the files interact, to be run with GNU Make.
- **`requirements.txt`** and **`pyproject.toml`**: Two most common formats to lists all python dependencies (and minimal versions, in the case of latter one).
- **`results/clustering_performance_results_formatted.csv`**: Contains the results of experimental evaluations, including performance metrics and comparisons across different implementations.

## How to Use
The easiest way to reproduce our results is to fork this repository and use Github actions to run our code.

### System Requirements
We tested with the following software versions. Higher versions are very likely to work, lower versions might work, but we cannot guarantee compatibility or identical outputs.

- Linux (we tested successfully on ubuntu-24.04 and arch-linux-6.12.8)
- 1 CPU, about 500 MB of disk space (mainly for the venv), 1 GB of memory
- Software
  - python 3.9.21 or higher
  - texlive's lualatex 1.18.0
  - GNU Make 4.4.1
  - optional: qpdf 11.9.1 to linearize the PDF paper

1. Clone the repository :
   ```bash
   git clone git@github.com:nikitaaveritchev/kmeans-Extension-Ptolemy.git
   cd kmeans-Extension-Ptolemy
   ```

2. Run the build script:
  ```bash
  make all
  ```
  It takes about an hour of (single CPU) runtime to run all experiments and build the final PDF.


## License

This project contains code under two licenses:

1. **CC BY 4.0**: All original contributions in this project are licensed under the Creative Commons Attribution 4.0 International License.
2. **MIT License**: Portions of this project are derived from [jjcordano's project](https://github.com/jjcordano/elkans_kmeans), which is licensed under the MIT license. These portions remain under the MIT license.

### Attribution
If using this project, please provide attribution for both:
- The original work by [jjcordano](https://github.com/jjcordano/elkans_kmeans) under the MIT license.
- This project under the CC BY 4.0 license.
