import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

colors = ["#313695", "#a50026", "#fee838", "#004529"]
status = ["susceptible", "exposed", "infectious", "recovered"]


def plot_node_degree(graph):
    """
    Plots the node degree distribution
    :param graph: The graph to analyze
    :return: None
    """
    deg = [x[1] for x in list(nx.degree(graph))]
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5, forward=True)
    sns.distplot(deg, kde=True)
    ax.set_xticks(np.arange(0, 24, 2))
    plt.show()


def plot_community_degree(graph):
    """
    Plots the community degree distribution
    :param graph: The graph to analyze
    :return: None
    """
    temp = nx.get_node_attributes(graph, 'community')
    es = []
    for k in temp:
        for x in temp[k]:
            if k != x:
                es.append((k, x))
    fams = graph.edge_subgraph(es)
    deg = [x[1] for x in list(nx.degree(fams))]
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5, forward=True)
    sns.distplot(deg, kde=True)
    ax.set_xticks(np.arange(0, 20, 1))
    plt.show()


def plot_family_degree(graph):
    """
    Plots the family degree distribution
    :param graph: The graph to analyze
    :return: None
    """
    temp = nx.get_node_attributes(graph, 'family')
    es = []
    for k in temp:
        for x in temp[k]:
            if k != x:
                es.append((k, x))
    fams = graph.edge_subgraph(es)
    deg = [x[1] for x in list(nx.degree(fams))]
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5, forward=True)
    sns.distplot(deg, kde=True)
    ax.set_xticks(np.arange(0, 10, 1))
    plt.show()


def plot_seir(sus, exp, inf, rec):
    """
    Plots a SEIR graph for simulation results
    :param sus: List of number of sus at time step {index+1}
    :param exp: List of number of exp at time step {index+1}
    :param inf: List of number of inf at time step {index+1}
    :param rec: List of number of rec at time step {index+1}
    :return: None
    """
    plt.figure(figsize=(18, 10))
    plt.plot(range(len(sus)), sus, c=colors[0])
    plt.plot(range(len(exp)), exp, c=colors[1])
    plt.plot(range(len(inf)), inf, c=colors[2])
    plt.plot(range(len(rec)), rec, c=colors[3])
    plt.legend(status)
    plt.title("SEIR graph for simulation that lasted {} days".format(len(sus)))
    plt.show()


def plot_graph(G):
    """
    Plots the graph G
    :param G: The graph to plot
    :return: None
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    pos = nx.spring_layout(G)
    for i in range(len(status)):
        node_subset = [x for x, y in G.nodes(data=True) if y["status"] == i]
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=node_subset, node_size=100,
                               node_color=colors[i], label=status[i], alpha=.75)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="#a2a2a2", alpha=.75)
    plt.legend()
