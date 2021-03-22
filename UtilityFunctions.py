import numpy as np
from scipy.integrate import odeint
import networkx as nx
import time
import random as rn
import itertools

t_se = 10.0
t_ei = 7.0
t_ir = 14.0
t_er = 14.0
t_rs = 60
p_E_to_R = 0.179
R_0_start = 3
k = 2.5
R_0_end = 0.9
x0 = 90
gamma = 1.0 / t_ir


def deriv(y, t, beta, N, t_ei, t_ir, t_er, t_rs, p_E_to_R):
    # Calculates the derivative, used for compartmental model with beta
    S, E, I, R = y

    dSdt = (R / t_rs) - (S * I * beta(t) / N)
    dEdt = (S * I * beta(t) / N) - (E / t_er) * p_E_to_R - (E / t_ei) * (1 - p_E_to_R)
    dIdt = (E / t_ei) * (1 - p_E_to_R) - I / t_ir
    dRdt = I / t_ir - R / t_rs
    return dSdt, dEdt, dIdt, dRdt


def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    # Gets logistic R_0, used for compartmental model
    return (R_0_start - R_0_end) / (1 + np.exp(-k * (-t + x0))) + R_0_end


def beta(t):
    # Calculates beta for given time, used for compartmental model
    return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma


def derivative(y, t, N, t_se, t_ei, t_ir, t_er, t_rs, p_E_to_R):
    # Calculates the derivative, used for compartmental model without beta
    S, E, I, R = y

    dSdt = (R / t_rs) - (S * I / N * t_se)
    dEdt = (S * I / N * t_se) - (E / t_er) * p_E_to_R - (E / t_ei) * (1 - p_E_to_R)
    dIdt = (E / t_ei) * (1 - p_E_to_R) - I / t_ir
    dRdt = I / t_ir - R / t_rs
    return dSdt, dEdt, dIdt, dRdt


def calculate_new_states_beta(days=1, exposed_0=5):
    # Calculates new states for compartmental model with beta
    # N = graph_size
    N = 1000
    y0 = N - exposed_0, exposed_0, 0.0, 0.0
    t = np.linspace(0, days - 1, days)
    ret = odeint(deriv, y0, t, args=(beta, N, t_ei, t_ir, t_er, t_rs, p_E_to_R))
    S, E, I, R = ret.T
    R_0_over_time = [beta(i) / gamma for i in range(len(t))]

    return t, S, E, I, R, R_0_over_time


def calculate_new_states(days=1, exposed_0=5):
    # Calculates new states for compartmental model without beta
    # N = graph_size
    N = 1000
    y0 = N - exposed_0, exposed_0, 0.0, 0.0
    t = np.linspace(0, days - 1, days)

    # Draw from distributions
    t_se = 10.0
    t_ei = 7.0
    t_ir = 14.0
    t_er = 14.0
    t_rs = 60
    p_E_to_R = 0.179

    ret = odeint(derivative, y0, t, args=(N, t_se, t_ei, t_ir, t_er, t_rs, p_E_to_R))
    S, E, I, R = ret.T

    return S, E, I, R


def generate_graph(num_nodes, tau1=2, tau2=5, mu=0.1, max_degree=15, average_degree=5,
                   min_community=6, max_community=30):
    """
    A function which generates a LFR benchmark graph with communities
    :param num_nodes: Number of nodes in the graph
    :param tau1: Tau1 parameter from LFR
    :param tau2: Tau2 parameter from LFR
    :param mu: Mu parameter from LFR
    :param max_degree: Maximum degree of a node
    :param average_degree: Average degree of a node
    :param min_community: Minimum number of nodes in a community
    :param max_community: Maxmimum number of nodes in a community
    :return: The graph G
    """
    G = nx.LFR_benchmark_graph(num_nodes, tau1, tau2, mu, average_degree, max_degree=max_degree,
                               min_community=min_community, max_community=max_community, seed=42)
    # Adding needed attributes used for graph model
    nx.set_node_attributes(G, 0, "status")
    nx.set_node_attributes(G, -1, "can_infect_again")
    nx.set_node_attributes(G, -1, "asymp")
    nx.set_node_attributes(G, 0, "time_in_state")
    return G


def add_families_to_graph(F, num_population):
    """
    A function which adds families to a graph F inplace
    :param F: The graph to which to add the families
    :param num_population: The population number for the graph
    :return: The graph with added families
    """
    # Used 2019 USA demographics to get these numbers
    num_family_members = [1, 2, 3, 4, 5, 6, 7]
    probabilities = [0.2837, 0.3451, 0.1507, 0.1276, 0.0578, 0.0226, 0.0125]

    family_total = 0
    families = []
    # Here we generate a list of number of family members
    while family_total < num_population:
        family_c = rn.choices(num_family_members, probabilities)[0]
        family_max = min(family_c, num_population - family_total)
        family_total += family_max
        families.append(family_max)
    n = 0
    # Populating the families
    for family_count in families:
        members_in_family = list(range(n, n + family_count, 1))
        edges = itertools.combinations(members_in_family, 2)
        # Connecting each family member in the graph
        F.add_edges_from(list(edges))
        for member in members_in_family:
            members_copy = members_in_family.copy()
            members_copy.remove(member)
            F.nodes[member]['family'] = members_copy
        n += family_count

    return F


def decision(probability):
    """
    Function which represents an event with probability equal to the probability argument
    :param probability: The probability for the event
    :return: True or False, whether the event happened or not
    """
    return rn.random() < probability


def get_param_list(family, num_days, set_initial_exposed, set_num_quarantines, set_quarantine_len,
                   set_t_ei, set_t_rs, set_reinfection, set_comm_talk_freq, set_comm_talk_freq_change,
                   set_other_talk_freq, set_other_talk_freq_change, set_p_er, set_gray_mass, beta=[3]):
    final_list = [[family], [num_days], set_initial_exposed, set_num_quarantines, set_quarantine_len,
                  set_t_ei, set_t_rs, set_reinfection, set_comm_talk_freq, set_comm_talk_freq_change,
                  set_other_talk_freq, set_other_talk_freq_change, set_p_er, set_gray_mass, beta]
    all_combinations = list(itertools.product(*final_list))
    list_dictionaries = []
    for combination in all_combinations:
        params = {
            "family": family,
            "num_days": num_days,
            "initial_exposed": combination[2],
            "beta": combination[-1],
            "num_quarantines": combination[3],
            "quarantine_len": combination[4],
            "t_ei": combination[5],
            "t_rs": combination[6],
            "reinfection": combination[7],
            "comm_talk_freq": combination[8],
            "comm_talk_freq_change": combination[9],
            "other_talk_freq": combination[10],
            "other_talk_freq_change": combination[11],
            "p_er": combination[12],
            "gray_mass": combination[13]
        }
        #params["num_quarantines"] = params["num_quarantines"][0]
        #params["quarantine_len"] = params["quarantine_len"][0]
        list_dictionaries.append(params)
    return list_dictionaries

