import networkx as nx
import numpy as np
import random as rd
import matplotlib.pyplot as plt


def k_distrib(graph, scale="lin", color="#40A6D1", alpha=0.8):
    """Creates a degree distribution plot for an arbitrary graph.

	Parameters
	----------
	graph : networkx.Graph
		The given graph providing the nodes
	scale : str, optional
		Indicate the graph type, either 'lin' (linear) or 'log' (logarithmic)
	colour : str, optional
		Hex code indicating plotting color
	alpha : int, optional
		Integer ranging within [0, 1] indicating the plotting opacity

	Reference
	----------
	Aleksander Molak
	https://github.com/AlxndrMlk/Barabasi-Albert_Network
	"""
    # initialization
    plt.close()
    num_nodes = graph.number_of_nodes()
    max_degree = 0
    x_axis = []
    y_axis_tmp = []

    # calculate the maximum degree (domain)
    for node in graph.nodes():
        if graph.degree(node) > max_degree:
            max_degree = graph.degree(node)

    # compute the portion of nodes for each degree (range)
    for i in range(max_degree + 1):
        x_axis.append(i)
        y_axis_tmp.append(0)
        for n in graph.nodes():
            if graph.degree(n) == i:
                y_axis_tmp[i] += 1
        y_axis = [i / num_nodes for i in y_axis_tmp]

    # customize plot scale (linear or logarithmic)
    if scale == "log":
        plt.title("Degree distribution (log-log scale)")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("log(k)")
        plt.ylabel("log(P(k))")
    else:  # scale will be considered as "lin"
        plt.xlabel("k")
        plt.ylabel("P(k)")

    plt.plot(
        x_axis, y_axis, linewidth=0, marker="o", markersize=8, color=color, alpha=alpha
    )
    plt.show()


def _get_random_node(graph: nx.Graph):
    node_probs = []
    for node in graph.nodes():
        node_prob = (graph.degree(node)) / (2 * len(graph.edges()))
        node_probs.append(node_prob)
    return np.random.choice(graph.nodes(), p=node_probs)


def _add_edge(graph: nx.Graph, new_node):
    if len(graph.edges()) == 0:
        random_node = 0
    else:
        random_node = _get_random_node(graph)

    new_edge = (random_node, new_node)
    if new_edge in graph.edges():
        _add_edge(graph, new_node)
    else:
        graph.add_edge(new_node, random_node)


def barabasi_albert_network(num_nodes_initial, num_nodes_final, m_parameter, graph):
    """Returns a Barabási-Albert scale-free network model

	Parameters
	----------
	num_nodes_initial : int
		Initial number of nodes in the network before applying the Barabási-Albert algorithm
	num_nodes_final : int
		Final number of nodes in the network after applying the Barabási-Albert algorithm
	m_parameter : int
		Number of edges to attach from a new node to existing nodes
	initial_graph : networkx.Graph, optional
		The initial graph to be used for the Barabási-Albert algorithm 

	Returns
	-------
	graph : networkx.Graph
	"""
    # initialization
    if graph is None:
        graph = nx.complete_graph(num_nodes_initial)

    index = 0
    new_node = num_nodes_initial

    for i in range(num_nodes_final - num_nodes_initial):
        # add_node() must take a unique hashable object to generate node identifier
        # hashable object is one that can be used as a key in a Python dictionary, e.g., numbers
        graph.add_node(num_nodes_initial + index)
        index += 1
        for j in range(0, m_parameter):
            _add_edge(graph, new_node)
        new_node += 1

    return graph
