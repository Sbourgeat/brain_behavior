# Create  new script to compare the bottleneck distances with the euclidean distances between behaviours in the DGRP lines

- Compare Riddha's behavioural scores delta with the bottleneck distances.
- Use the previous behaviour comparison code to test on delta brains scores.
    - Use bayesian model to model the distribution of the behaviour
    - Calculate the entropy of each distributions
    - Calculate the bottleneck distances between the brains
    - compare the bottleneck distances and the KL divergence between two DGRP lines
    - Do it for all the DGRP lines in a pairwise manner and get the correlation coefficient in the matrix
    - And maybe get a p value for the correlation coefficient