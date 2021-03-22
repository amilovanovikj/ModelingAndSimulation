# -*- coding: utf-8 -*-
from GraphVisualizations import *
from UtilityFunctions import *
import random as rn
import numpy as np
import scipy
from copy import deepcopy
import plotly.graph_objects as go
from plotly.offline import plot
from IPython.core.display import display, HTML
config={'showLink': False, 'displayModeBar': False}


def run_simulation(G, params, show_seir=True, show_graph=False, ret_seir=False):
    """
    This function runs a graph based SEIRS simulation model
    :param G: The graph on which to run the simulation
    :param params: A dictionary of hyperparameters to be tested
    :param show_seir: Whether to plot the SEIRS plot
    :param show_graph: Whether to show the graph after the simulation has finished
    :param ret_seir: Whether to return the simulation values, only used by slider simulation function
    :return: Can return the Seir values, or None
    """
    # Copying graph in order to not make inline changes
    Graph = deepcopy(G)
    num_nodes = len(G.nodes)
    # Adding initial exposure according to the params value
    initial_exposed = params["initial_exposed"]
    # Defining all lists of states
    exposed = rn.sample(range(0, num_nodes - 1), initial_exposed)
    sus = list(set(range(0, num_nodes, 1)).difference(set(exposed)))
    infected = []
    recovered = []

    for i in exposed:
        # Switching state status for the exposed nodes
        Graph.nodes[i]["status"] = 1

    # Keeping track of number of nodes in each state over time to plot later on
    S = [num_nodes - initial_exposed]
    E = [initial_exposed]
    I = [0]
    R = [0]

    # Copying hyperparameters from the params dictionary
    t_ei = params["t_ei"]  # Average time in time units(day) to switch from exposed to infected
    t_rs = params["t_rs"]  # Average time in time units(day) to switch from recovered to susceptible
    reinfection = params["reinfection"]  # Probability that a recovered individual can reinfect again, only decided once
    # per node
    comm_talk_freq = params["comm_talk_freq"]  # The frequency with which a node communicates with nodes in the
    # community. More precisely, the number defines the average span in which the node communicates with every other
    # node in the community. For example if comm_talk_freq=3, that means that every node communicates once every 3 days
    # with every other node in its community (in random order)
    comm_talk_freq_change = params[
        "comm_talk_freq_change"]  # Increment/decrement to be used for the comm_talk_freq when
    # inducing quarantines
    other_talk_freq = params["other_talk_freq"]  # The frequency with which a node communicates with nodes
    # that aren't in its family or community analogous to the comm_talk_freq
    other_talk_freq_change = params["other_talk_freq_change"]  # Increment/decrement to be used for the other_talk_freq
    # when inducing quarantines
    p_E_to_R = params["p_er"]  # The probability of switching state from exposed to recovered (only if person is
    # asymptomatic)
    gray_mass = params[
        "gray_mass"]  # Probability that a person will get exposed no matter what happens at a certain time step due to
    # unknown reasons
    quarantine_nums = params["num_quarantines"]  # Number of quarantines in the time span of the simulation to be
    # generated
    quarantine_len = params["quarantine_len"]  # The quarantine length
    num_days = params["num_days"]  # Number of days the simulation lasts
    families = params["family"]  # Whether the graph has families or not
    beta = params["beta"]  # The beta value (explained below)

    # Dictionary to convert from status to number
    status = {"sus": 0, "exp": 1, "inf": 2, "rec": 3}

    # This section controls the quarantines(dynamic changes) if added as hyperparameters
    quarantine_cur_day = 0
    if quarantine_nums > 0:
        # This is the first day the quarantine starts. The quarantine also repeats itself every i*period_out days where
        # i = 1,2....etc
        period_out = int(num_days / quarantine_nums) - quarantine_len
    for i in range(num_days):
        if quarantine_nums > 0:
            if 0 < quarantine_cur_day < quarantine_len + 2:
                # Already in quarantine, communication reduced
                if quarantine_cur_day < 4:
                    # During the first 3 days of a quarantine we gradually decrease the communication between
                    # communities and other nodes
                    comm_talk_freq += comm_talk_freq_change
                    other_talk_freq += other_talk_freq_change
                elif quarantine_cur_day > quarantine_len - 2:
                    # During the last 3 days of a quarantine we gradually increase the communication between communities
                    # and other nodes back to normal
                    comm_talk_freq -= comm_talk_freq_change
                    comm_talk_freq = max(0, comm_talk_freq)
                    other_talk_freq -= other_talk_freq_change
                    other_talk_freq = max(0, other_talk_freq)
                # Day finished
                quarantine_cur_day += 1
            else:
                # Quarantine finished, reset variables
                comm_talk_freq = params["comm_talk_freq"]
                other_talk_freq = params["other_talk_freq"]
                quarantine_cur_day = 0
            if quarantine_nums > 0 and (i + 1) % period_out == 0:
                # Quarantine start
                quarantine_cur_day += 1

        # This section controls probability based state shifts for every node in the graph in a single time unit (day)
        for j in range(num_nodes):
            Graph.nodes[j]["time_in_state"] += 1
            # Recovered to susceptible state change
            # A recovered person can only transition to susceptible
            sus_now = False  # Variables of these type make sure that we don't induce 2 state changes for a single node
            # in a single day
            if Graph.nodes[j]["status"] == status["rec"]:
                # If can_infect_again = 1, that means that the person was already checked for reinfection and it was
                # concluded that he can reinfect again so we check for the transition
                if Graph.nodes[j]["can_infect_again"] == 1:
                    # If yes, then we test if the recovered person can transition to sus with the parameter t_rs
                    p_r_to_s = 1.0 / t_rs
                    choice = decision(p_r_to_s)
                    if choice:
                        # If the event happened, we change the state of the node
                        Graph.nodes[j]["status"] = status["sus"]
                        Graph.nodes[j]["time_in_state"] = 0  # reseting time in state
                        recovered.remove(j)
                        sus.append(j)
                        sus_now = True
                        Graph.nodes[j]["can_infect_again"] = -1  # We also set this variable to be -1 in order for the
                        # node to be applicable to be tested again in the future for possible reinfection
                elif Graph.nodes[j]["can_infect_again"] == -1:
                    # If we still haven't decided for the given node whether it can reinfect or not, we decide
                    choice = decision(reinfection)
                    if choice:
                        # Meaning the node can reinfect
                        # Decide prob of transition
                        inner_choice = decision(1 / t_rs)
                        if inner_choice:
                            # Transitioning to sus
                            sus_now = True
                            Graph.nodes[j]["status"] = status["sus"]
                            Graph.nodes[j]["time_in_state"] = 0
                            Graph.nodes[j]["can_infect_again"] = -1
                            recovered.remove(j)
                            sus.append(j)
                        else:
                            # Flag for check
                            Graph.nodes[j]["can_infect_again"] = 1
                    else:
                        # The node can't reinfect so we just keep it in the recovered state
                        Graph.nodes[j]["can_infect_again"] = 0
            # Susceptible to exposed state change A susceptible person can only transition to exposed. Factors which
            # affect this transition are the rate of contact, beta = β (“beta”), the expected amount of people an
            # infected person infects per day and the number of contacts. The beta can be used to derive the
            # probability that a random individiual will get infected when contacting an infected person. This
            # probability = beta/N^ where N^ is equal to the average number of contacts the infected person sees per
            # day (or his rate of contact). This means that the total probability of transition is 1 -
            # product_i:infected_neighbors(1-beta/N^_i). (1-beta/N^_i) is the probability that infected person i
            # which you are in contact with and see doesn't infect you. The product is the probability that no one of
            # the infected people you meet that day infects you. 1-this is the probability that at least one of them
            # infects you. We need to integrate here knowledge about whether the contacts are in a family,
            # in a community or other contacts in order to see the rate of communication between the target node and
            # them. So, out of all infected neighbors of node j, we always take into account family members,
            # we take into account community members every comm_talk_freq days (1/comm_talk_freq chance that j will
            # see a community friend) and every other contact we see every other_talk_freq days(1/other_talk_freq
            # chance that j will see something else(not in community nor family)). The chance to see a community
            # person who is infected is 1/(comm_talk_freq) * number of infected community neighbors/all neighbors.
            # Same as other members. We use this to derive the final formula : 1-[(1-beta/N_1..*)^family_count)* (
            # 1-beta/N_1..* + freq_community-1/freq_community)^community_count * (1-beta/N_1..* +
            # freq_other-1/freq_other)^other_count
            exp_now = False
            if Graph.nodes[j]["status"] == status["sus"] and not sus_now:
                if not exp_now:
                    neighbors = [n for n in Graph.neighbors(j)]
                    family = []
                    if families:
                        # Get all family members for the node
                        family = [neighbor for neighbor in neighbors if
                                  neighbor in Graph.nodes[j]["family"]]
                    community = list(Graph.nodes[j]["community"])
                    # Get all community members for the node
                    community_final = [n for n in community if n in neighbors and n not in family]
                    if j in community_final:
                        community_final.remove(j)  # Remove itself
                    # Get all other members for node (not family and not community but neighbors, usually a small
                    # number because of the graph generation process)
                    other = [neighbor for neighbor in neighbors if
                             neighbor not in family and neighbor not in community_final]
                    if j in other:
                        other.remove(j)  # Remove itself if inside

                    # This mulitplier will represent the probability that no one from all the infected nodes the target
                    # node knows and sees, will infect them (the product of 1-beta/N^ for all infected neighbors)
                    multiplier = 1  # Starting from 1, meaning 0 change to infect
                    for family_member in family:
                        # Now we check for each family member of the node, which of them is infected
                        if Graph.nodes[family_member]["status"] == status["inf"]:  # if family node is infected
                            # Since a family node is infected, we need to analyze its connections and how many nodes
                            # in average this family members communicates with on a daily basis
                            neighbors_in = [n for n in Graph.neighbors(family_member)]
                            family_in = []
                            if families:
                                family_in = [neighbor for neighbor in neighbors_in if
                                             neighbor in Graph.nodes[family_member]["family"]]
                            community_in = list(Graph.nodes[family_member]["community"])
                            community_final_in = [n for n in community_in if n in neighbors_in and n not in family_in]
                            if family_member in community_final_in:
                                community_final_in.remove(family_member)  # Removing itself
                            other_in = [n for n in neighbors_in if n not in family_in and n not in community_final_in]
                            if family_member in other_in:
                                other_in.remove(family_member)  # Removing itself

                            # Now we find the average people this family node sees per day
                            average_people_met = len(family_in)
                            if comm_talk_freq > 0:
                                average_people_met += (float(len(community_final_in)) / comm_talk_freq)
                            if other_talk_freq > 0:
                                average_people_met += (float(len(other_in)) / other_talk_freq)

                            if average_people_met == 0:
                                multiplier *= (1 - beta)
                            else:
                                if beta / average_people_met <= 1:
                                    # Now this term (1 - beta / average_people_met) gets added to the final product and
                                    # represents the probability that this family member won't infect the node j on this
                                    # day
                                    multiplier *= (1 - beta / average_people_met)
                    for community_member in community_final:
                        # We do the same for every infected community member also
                        if Graph.nodes[community_member]["status"] == status["inf"]:  # if community infected
                            neighbors_in = [n for n in Graph.neighbors(community_member)]
                            family_in = []
                            if families:
                                family_in = [neighbor for neighbor in neighbors_in if
                                             neighbor in Graph.nodes[community_member]["family"]]
                            community_in = list(Graph.nodes[community_member]["community"])
                            community_final_in = [n for n in community_in if n in neighbors_in and n not in family_in]
                            if community_member in community_final_in:
                                community_final_in.remove(community_member)  # Removing itself
                            other_in = [n for n in neighbors_in if n not in family_in and n not in community_final_in]
                            if community_member in other_in:
                                other_in.remove(community_member)  # Removing itself
                            average_people_met = len(family_in)
                            if comm_talk_freq > 0:
                                average_people_met += (float(len(community_final_in)) / comm_talk_freq)
                            if other_talk_freq > 0:
                                average_people_met += (float(len(other_in)) / other_talk_freq)
                            if average_people_met == 0:
                                if comm_talk_freq == 0:
                                    multiplier = 1
                                else:
                                    if beta / comm_talk_freq <= 1:
                                        multiplier *= (1 - beta / comm_talk_freq)
                            else:
                                if comm_talk_freq == 0:
                                    if beta / average_people_met <= 1:
                                        multiplier *= (1 - beta / average_people_met)
                                else:
                                    if beta / (average_people_met * comm_talk_freq) <= 1:
                                        # Here we also add the parameter comm_talk_freq in order to integrate the
                                        # knowledge that the infected person is in our community so we only see him
                                        # every comm_talk_freq days
                                        multiplier *= (1 - beta / (average_people_met * comm_talk_freq))
                    for other_member in other:
                        # Same for all other members
                        if Graph.nodes[other_member]["status"] == status["inf"]:  # if other infected
                            neighbors_in = [n for n in Graph.neighbors(other_member)]
                            family_in = []
                            if families:
                                family_in = [neighbor for neighbor in neighbors_in if
                                             neighbor in Graph.nodes[other_member]["family"]]
                            community_in = list(Graph.nodes[other_member]["community"])
                            community_final_in = [n for n in community_in if n in neighbors_in and n not in family_in]
                            if other_member in community_final_in:
                                community_final_in.remove(other_member)  # Removing itself
                            other_in = [n for n in neighbors_in if n not in family_in and n not in community_final_in]
                            if other_member in other_in:
                                other_in.remove(other_member)  # Removing itself
                            average_people_met = len(family_in)
                            if comm_talk_freq > 0:
                                average_people_met += (float(len(community_final_in)) / comm_talk_freq)
                            if other_talk_freq > 0:
                                average_people_met += (float(len(other_in)) / other_talk_freq)
                            if average_people_met == 0:
                                if other_talk_freq == 0:
                                    multiplier = 1
                                else:
                                    if beta / other_talk_freq <= 1:
                                        multiplier *= (1 - beta / other_talk_freq)
                            else:
                                if other_talk_freq == 0:
                                    if beta / average_people_met <= 1:
                                        multiplier *= (1 - beta / average_people_met)
                                else:
                                    if beta / (average_people_met * other_talk_freq) <= 1:
                                        # Here we use the other talk freq
                                        multiplier *= (1 - beta / (average_people_met * other_talk_freq))

                    # The final probability of infection is 1-multipler+gray_mass, because the multipler represents
                    # the probability that not a single infected person we know will infect us that day, so we just look
                    # for the opposite event. The gray mass is always added as a flat probability
                    probability_of_infection = 1 - multiplier + gray_mass
                    # It is important to note that this probability will never exceed 1+gray_mass

                    choice = decision(probability_of_infection)
                    if choice:
                        # If the person is exposed we switch his status
                        Graph.nodes[j]["status"] = status["exp"]
                        Graph.nodes[j]["time_in_state"] = 0
                        sus.remove(j)
                        exposed.append(j)
                        exp_now = True
            # Exposed to infected/recovered state change
            # An exposed person can either become infected(infectious) in t_ei days or
            # become asympt in t_er days. The probability an exposed person becomes asympt at day 1 of exposure is
            # p_E_to_R (decided at day 1 for every person - the next day after exposure/ during this day the person
            # can't infect anyone). This means that for every node we first have to decide whether or not the exposed
            # node will be asympt or not. If it is then, the probability of transitioning to recovered is 1/t_er. If
            # it isn't, the probability of transitioning to infected is 1/t_ei.
            inf_now = False
            if Graph.nodes[j]["status"] == status["exp"] and not exp_now:
                if Graph.nodes[j]["asymp"] != -1:
                    # This means we have already decided whether or not the person is asympt.
                    if Graph.nodes[j]["asymp"] == 1:  # asympt
                        # We sample the probability to go from exp to rec from neg binom (derived from paper)
                        sample = scipy.stats.nbinom.pmf(Graph.nodes[j]["time_in_state"], 6, 0.35)
                        choice = decision(sample)
                        if choice:
                            # State change
                            Graph.nodes[j]["status"] = status["rec"]
                            Graph.nodes[j]["time_in_state"] = 0
                            exposed.remove(j)
                            recovered.append(j)
                    else:
                        # The person is not asympt so he will infect and spread the virus, regulated by t_ei
                        choice = decision(1.0 / t_ei)
                        if choice:
                            Graph.nodes[j]["status"] = status["inf"]
                            Graph.nodes[j]["time_in_state"] = 0
                            exposed.remove(j)
                            infected.append(j)
                            inf_now = True
                else:  # Haven't decided whether the person is asympt or not
                    choice = decision(p_E_to_R)  # Deciding asympt
                    if choice:
                        Graph.nodes[j]["asymp"] = 1  # asympt
                        # Neg bin again
                        sample = scipy.stats.nbinom.pmf(Graph.nodes[j]["time_in_state"], 6, 0.35)
                        choice = decision(sample)
                        if choice:
                            Graph.nodes[j]["status"] = status["rec"]
                            Graph.nodes[j]["time_in_state"] = 0
                            exposed.remove(j)
                            recovered.append(j)
                    else:  # not asympt
                        Graph.nodes[j]["asymp"] = 0
                        choice = decision(1.0 / t_ei)
                        if choice:
                            Graph.nodes[j]["status"] = status["inf"]
                            Graph.nodes[j]["time_in_state"] = 0
                            exposed.remove(j)
                            infected.append(j)
                            inf_now = True
            # Infected to recovered state change
            # An infected person can only recover. An infected person recovers after with a neg binom dist
            if Graph.nodes[j]["status"] == status["inf"] and not inf_now:
                sample = scipy.stats.nbinom.pmf(Graph.nodes[j]["time_in_state"], 6, 0.35)
                choice = decision(sample)
                if choice:
                    Graph.nodes[j]["status"] = status["rec"]
                    Graph.nodes[j]["time_in_state"] = 0
                    infected.remove(j)
                    recovered.append(j)

        # Finally appending the values for S,E,I,R after day i
        S.append(num_nodes - len(exposed) - len(infected) - len(recovered))
        E.append(len(exposed))
        I.append(len(infected))
        R.append(len(recovered))

    if show_seir:
        plot_seir(S, E, I, R)
    if show_graph:
        plot_graph(Graph)
    if ret_seir:
        return [S, E, I, R]


def slider_simulation(G, params, slide_by='beta', values=np.arange(1, 3, 0.5), show_graph=False):
    """
    This function makes a slider simulation, where the slider variable is the defined by the slide_by parameter. This
    allows for ablation studies and interesting comparion plots for certain parameters(for example beta, number of
    quarantines etc..)
    :param G: The graph to plot for
    :param params: The params dictionary
    :param slide_by: The variable to be used for the slider
    :param values: The values of the slider variable to plot for
    :param show_graph: Whether to show the graph
    :return: None
    """
    num_days = params["num_days"]
    fig = go.Figure(layout=dict(title="SEIR model, number of days: {}, {}: {} ".format(num_days, slide_by, values[0])))
    x = list(range(0, num_days + 1))
    for v in values:
        params[slide_by] = v
        seir = run_simulation(G, params, show_graph=show_graph, show_seir=False, ret_seir=True)
        for i, s in enumerate(seir):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(width=1, color=colors[i]),
                    name=status[i],
                    x=x,
                    y=s
                )
            )
    for j in range(4):
        fig.data[j].visible = True
    steps = []
    for i in range(len(values)):
        step = dict(
            method="update",
            label="{}".format(values[i]),
            args=[{"visible": [False] * len(fig.data) * 4},
                  {"title": "SEIR model, number of days: {}, {}: {} ".format(num_days, slide_by, values[i])}],
        )
        for j in range(4):
            step["args"][0]["visible"][4 * i + j] = True
        steps.append(step)
    sliders = [dict(
        active=0,
        currentvalue={"prefix": f"{slide_by}: "},
        pad={"t": 50},
        steps=steps)]
    fig.update_layout(sliders=sliders, template="plotly_white", showlegend=True, width=750, height=575)
    plot(fig, filename = 'fig.html', config = config)
    display(HTML('fig.html'))
