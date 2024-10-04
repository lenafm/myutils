# generate networks

import graph_tool.all as gt
import numpy as np


def get_block_graph(g: gt.Graph, b: gt.VertexPropertyMap, metadata: gt.VertexPropertyMap = None) -> gt.Graph:
    """
    Get the block graph induced by a partition. Optionally, compute the mean of a continuous 'metadata' vertex
    property for each supernode if the 'metadata' parameter is provided.

    Args:
        g (gt.Graph): The input graph.
        b (gt.VertexPropertyMap): The vertex property map representing the partition.
        metadata (gt.VertexPropertyMap, optional): The continuous vertex property map to compute the average
            value for each supernode. Defaults to None.

    Returns:
        gt.Graph: The block graph induced by the partition, with the average metadata value for each supernode
        if the metadata is provided.

    Raises:
        ValueError: If metadata is provided but is not continuous.
    """
    # Condense the graph using the provided partition
    cg, br, vc, ec, av, ae = gt.condensation_graph(g, b, avprops=[metadata], self_loops=True)

    # Assign vertex and edge counts
    cg.vp.count = vc
    cg.ep.count = ec

    cg.vp.mean_metadata = cg.new_vertex_property('double')
    for v in cg.vertices():
        cg.vp.mean_metadata[v] = av[0][v] / vc[v]

    return cg


def generate_network(N, B, mu, mean_degree, equal_block_sizes, micro_ers=False, mesotype='communities',
                     sizes=None, seed=None):
    M = create_block_matrix(mu=mu, N=N, B=B, mean_degree=mean_degree, mesotype=mesotype)
    b = create_block_membership_vector(N=N, B=B, equal_block_sizes=equal_block_sizes, sizes=sizes)
    if seed is not None:
        gt.seed_rng(seed)
    g = gt.generate_sbm(b=b, probs=M, micro_ers=micro_ers)
    g.vp.blocklabel = g.new_vp("int", vals=b)
    g = gt.GraphView(g, vfilt=gt.label_largest_component(g, directed=False))
    return gt.Graph(g, prune=True)


def create_block_matrix(mu: float, N: int, B: int, mean_degree: float, mesotype: str) -> np.ndarray:
    """
    Create a block matrix for a Stochastic Blockmodel.

    Args:
        mu (float): Proportion of intra-community edges.
        N (int): Total number of vertices.
        B (int): Number of communities or blocks.
        mean_degree (float): Mean degree of the graph.
        mesotype (str): Type of mesoscale structure. Should be either "communities" or "cp" (core-periphery).

    Returns:
        np.ndarray: Block matrix representing the connectivity pattern in the Stochastic Blockmodel.

    Raises:
        ValueError: If mesotype is neither "communities" nor "cp" (core-periphery).

    """
    c = 1 - mu
    E = mean_degree * N / 2
    M = np.zeros((B, B))
    if mesotype == 'communities':
        for i in range(B):
            for j in range(B):
                if i == j:
                    m_ij = c / B
                else:
                    m_ij = (1 - c) / (B * (B - 1))
                M[i, j] = 2 * E * m_ij
    elif mesotype == 'cp':
        for i in range(B):
            for j in range(B):
                if i == j == 0:
                    m_ij = c / B
                elif i == j:
                    m_ij = (1 - c) / B
                else:
                    m_ij = 0.5 / (B * (B - 1))
                M[i, j] = 2 * E * m_ij
    else:
        raise ValueError('Mesotype must be either "communities" or "cp" (core-periphery).')
    return np.around(M,0).astype(int)


def create_block_membership_vector(N: int, B: int,
                                   equal_block_sizes: bool, sizes: np.ndarray = None) -> np.ndarray:
    """
    Creates a vector of length N where each element indicates the block membership of a node.

    Args:
    - N (int): the number of nodes
    - B (int): the number of blocks in this partition
    - equal_block_sizes (bool): whether the blocks should have equal sizes
    - sizes (np.ndarray): if not None, specifies the proportion of nodes assigned to each block

    Returns:
    - A numpy array of shape (N,) containing integers between 0 and B-1, representing the block membership of each
    node
    """
    if equal_block_sizes:
        assert N % B == 0, 'N must be a multiple of B if `equal_block_sizes = True`.'
        if sizes is not None:
            print('Ignoring `sizes` parameter as `equal_block_sizes = True`')
        x = np.repeat(np.arange(B), int(N/B))
        np.random.shuffle(x)
        return x
    else:
        if sizes is None:
            sizes = np.array([1/B] * B)
        else:
            assert np.isclose(np.sum(sizes), 1), 'The elements of `sizes` must sum to 1.'
        return np.random.choice(B, N, p=sizes)