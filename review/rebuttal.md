Thank you for your reviews! Let me address your questions:

Total Runtime Measurements (R2, R3)
Our paper only considers the number of explicit distance evaluations to measure the efficiency of the algorithms we compare. Reviews 2 and 3 bring up the valid point of missing total execution time measurements. While we mention this fact as a limitation (in ch. 4.3, Limitations, point ii), our paper does not give a good explanation for this choice.

We chose to count distance evaluations because they are agnostic to the specific implementation of algorithms in machine code. The main disadvantage of reporting execution times is their strong dependence on the amount of optimization that went into the implementation.
For example, time measurements are affected by the programming language chosen, the amount of time spend on optimizing the source code, and the specific choice of hardware architecture (e.g. GPU or CPU).
Therefore, it takes considerable effort and care to make time measurements objective, reproducible, and generalizable. Consequently, in our short paper we focus on the theoretical, implementation-independent aspects of the discussed algorithms.

That said, the actual time measurements are important for real-world applications: Besides the distance evaluations, the caching and evaluation of the upper and lower bound approximations also takes time. While we reduce the total number of distance computations compared to Elkan's algorithm, our new bounds require more memory accesses and computations (cf. Eq. 3,4 and Eq. 5,6).
While this additional time is likely to be small compared to the time spent on distance evaluations, we plan to measure and report the actual ratio in a future full paper.

For the camera-ready version of our paper, we will include this elaboration and explicitly add execution time measurements as future work.


Discussion of Observed Behavior.
Review 3 notes that our discussion does not give a clear indication of how the chosen parameters relate to the speedups achieved, i.e. how the number of clusters $k$, the dataset, and its dimensionality influence the number of distance calculations saved. We agree that more precise rules predicting the actual speedups would be immensely useful.

However, we believe that we cannot, in good conscience, derive better rules from our data than those presented in the paper: While we did not cherry-pick the datasets used in our investigation, we support Review 3's notion that the speedups appear to be dataset-dependent (ch. 4.3, Limitations, point i). Similarly, we believe that a wider range of starting parameters (initializations, $k$s) is needed to draw more robust conclusions (ch. 4.3, Limitations, point iii).

Consequently, we feel confident that our discussion cannot go into more detail without running the significant risk of arriving at non-generalizable conclusions.


Analysis of the Set of Distance Computations.
Review 3 suggests an analysis of which distances are computed in each algorithm. We think this is an idea worth pursuing in the preparation of a full paper, but the actual implementation of this approach seems to require considerable effort.
While obtaining the data needed for such an analysis is straightforward, how to interpret it is still unclear to us at this time. Clusterings on data beyond the two-dimensional case are difficult to interpret visually, making it harder to develop meaningful measures for comparing the distance computation sets.


Minor Problems.
Review 3 correctly notes that Eq. 2 has a typo in the first summand of the RHS. We will correct this mistake in the camera-ready version. We will also improve the readability and accessibility of Fig. 1 by increasing its font size and using different line styles rather than relying on coloring alone.


Review 1 does not contain any questions or statements that need to be rebutted.
