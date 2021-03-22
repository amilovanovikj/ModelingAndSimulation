from CovidSimulation import *
from copy import deepcopy

if __name__ == "__main__":
    # This script is used to test out the graph based model on a graph with or without families. All the parameters
    # important in the simulation should be provided in the params dictionary and passed to either the run simulation
    # function or the slider simulation function
    num_nodes = 301
    rn.seed(42)
    params = {
        "family": True,
        "num_days": 500,
        "initial_exposed": 5,
        "beta": 3,
        "num_quarantines": 15,
        "quarantine_len": 14,
        "t_ei": 8,
        "t_rs": 150,
        "reinfection": 0.17,
        "comm_talk_freq": 6,
        "comm_talk_freq_change": 2,
        "other_talk_freq": 10,
        "other_talk_freq_change": 2,
        "p_er": 0.2,
        "gray_mass": 0.0001
    }
    # Creating the graph
    G = generate_graph(num_nodes)

    # plot_graph(G)
    # plot_node_degree(G)
    print("The number of connected components in the graph is {}".format(len(list(nx.connected_components(G)))))
    print("The diameter of the graph is {}".format(nx.algorithms.distance_measures.diameter(G)))
    print("The average clustering coefficient of the graph is {}".format(nx.algorithms.cluster.average_clustering(G)))
    G_copy = deepcopy(G)
    # plot_graph(G)

    # Add families
    F = add_families_to_graph(deepcopy(G_copy), num_nodes)
    # plot_graph(G_copy)
    # plot_graph(F)

    # plot_node_degree(F)
    # plot_family_degree(F)
    # plot_community_degree(F)
    print("The number of connected components in the graph is {}".format(len(list(nx.connected_components(G)))))
    print("The diameter of the graph is {}".format(nx.algorithms.distance_measures.diameter(G)))
    print("The average clustering coefficient of the graph is {}".format(nx.algorithms.cluster.average_clustering(G)))
    rn.seed(1)
    # run_simulation(F, params=params, show_graph=True)
    slider_simulation(F, params, slide_by="num_quarantines", values=[0, 12, 14])
