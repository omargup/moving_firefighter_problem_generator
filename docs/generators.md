# Instance Generators

This document provides a detailed explanation of how each instance generator in the `movingfp` package works.

## Erdos Connected (`erdos_connected`)

This generator creates a connected graph based on the Erdős-Rényi model, which does not guarantee connectivity on its own. The generation process is as follows:

1.  **Graph Generation:** It iteratively generates random graphs using the Erdős-Rényi model with `n'` nodes, where `n <= n' < n + ceil(n/4)`. Each possible edge is added with probability `p`.
2.  **Connectivity Check:** After generating a graph, it searches for a connected component of size exactly `n`.
3.  **Selection:** If such a component is found, it is selected as the `G_fire` for the instance. If not, the process repeats until a suitable connected graph is found.
4.  **Node Placement:** The `n` nodes of the final graph are assigned positions chosen uniformly at random within the unit hypercube `[0,1]^d`.
5.  **Firefighter Graph:** The `G_fighter` is constructed as a complete graph. The weight of each edge is the Euclidean distance between the two nodes it connects (including the firefighter's position).

This process ensures that the resulting fire graph is connected, which is a common requirement for firefighter problems.