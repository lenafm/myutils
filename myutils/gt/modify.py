# filter graph object
import graph_tool.all as gt
import networkx as nx


def remove_isolated_nodes(g, prune=False):
    u = g.copy()
    u = gt.GraphView(u, vfilt=lambda v: v.in_degree() + v.out_degree() > 0)
    return gt.Graph(u, prune=prune)


def extract_lcc(g, directed=False, prune=True):
    g = gt.GraphView(g, vfilt=gt.label_largest_component(g, directed=directed))
    return gt.Graph(g, prune=prune)


def extract_retweet_graph(g, prune=False, lcc=False, twocore=False):
    return extract_tweet_type_graph(g=g, retweet_type_idx=0, prune=prune, lcc=lcc, twocore=twocore)


def extract_reply_graph(g, prune=False, lcc=False, twocore=False):
    return extract_tweet_type_graph(g=g, retweet_type_idx=1, prune=prune, lcc=lcc, twocore=twocore)


def extract_quote_graph(g, prune=False, lcc=False, twocore=False):
    return extract_tweet_type_graph(g=g, retweet_type_idx=2, prune=prune, lcc=lcc, twocore=twocore)


def extract_tweet_type_graph(g, retweet_type_idx, prune=False, lcc=False, twocore=False):
    if g.num_vertices() == 0:
        return gt.Graph(directed=True)
    assert ('e', 'tweet_type') in g.properties, 'Graph has no edge property "tweet_type".'
    u = gt.GraphView(g, efilt=lambda v: g.ep.tweet_type[v] == retweet_type_idx)
    if lcc:
        u = extract_lcc(u, prune=prune)
    if twocore:
        return extract_kcore(u, k=2, prune=prune)
    if prune:
        return gt.Graph(u, prune=True)
    return u


def extract_kcore(g, k, prune=True):
    kcore = gt.kcore_decomposition(g)
    vfilt = kcore.a > (k-1)
    g = gt.GraphView(g, vfilt=vfilt)
    return gt.Graph(g, prune=prune)


def is_multigraph(g):
    """
    Check whether a network is a multigraph or not.

    Args:
        graph: A graph_tool.Graph object.

    Returns:
        True if the graph is a multigraph, False otherwise.
    """
    edge_counts = {}

    for edge in g.edges():
        source = int(edge.source())
        target = int(edge.target())

        if (source, target) in edge_counts:
            edge_counts[(source, target)] += 1
        else:
            edge_counts[(source, target)] = 1

    return any(count > 1 for count in edge_counts.values())


def simplify_multigraph(multigraph):
    """
    Convert an undirected multigraph into a simple graph by removing duplicate edges.
    Preserve vertex properties from the input graph.

    Args:
        multigraph: A graph_tool.Graph object representing the undirected multigraph.

    Returns:
        A new graph_tool.Graph object representing the simplified graph with preserved vertex properties.
    """
    simple_graph = gt.Graph(directed=False)
    node_dict = {}  # Mapping of original node ids to new node ids

    # Copy vertex properties from the input multigraph to the simplified graph
    for prop_name, prop_value in multigraph.vp.items():
        simple_graph.vp[prop_name] = simple_graph.new_vertex_property(prop_value.value_type())

    for v in multigraph.vertices():
        new_v = simple_graph.add_vertex()

        for prop_name, prop_value in multigraph.vp.items():
            simple_graph.vp[prop_name][new_v] = prop_value[v]

        node_dict[int(v)] = new_v

    for edge in multigraph.edges():
        source = int(edge.source())
        target = int(edge.target())
        new_source = node_dict[source]
        new_target = node_dict[target]

        # Add the edge to the new graph if it doesn't exist already
        if not simple_graph.edge(new_source, new_target):
            simple_graph.add_edge(new_source, new_target)

    return simple_graph


def gt2nx(g):
    """
    Convert a GraphTool graph to a NetworkX graph.

    Args:
        graph_tool_graph: The GraphTool graph to be converted.

    Returns:
        A NetworkX graph equivalent to the input GraphTool graph.
    """
    # Create an empty NetworkX graph
    networkx_graph = nx.Graph()

    # Copy nodes from GraphTool to NetworkX
    for v in g.vertices():
        networkx_graph.add_node(int(v))

    # Copy edges from GraphTool to NetworkX
    for e in g.edges():
        source = int(e.source())
        target = int(e.target())
        networkx_graph.add_edge(source, target)

    return networkx_graph
