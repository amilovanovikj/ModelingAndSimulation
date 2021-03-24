# Introduction

In this work we present numerous simulations done on a complex network using a SEIRS model for COVID-19 spread. The network contains edges that represent one of three different types of connections between nodes (people) in the network: family, friends and acquaintances. The construction of the network, regarding the friend and acquaintance type of connection, is done using the LFR benchmark algorithm, and further extended to include family connections. It should be noted that each node is in contact with each family member on daily basis, while the contacts with friends and acquaintances is controlled by a parameter in the simulation (a node contacts its friends more often than its acquaintances). The simulations done on these networks include a large number of parameters:
- "family": boolean that indicates whether families are added to the graph and if the simulation should include them or not
- "num_days": integer specifying how many iterations (days) should the simulation last
- "initial_exposed": number of initially exposed nodes
- "beta": the infection rate
- "num_quarantines": integer that indicates how many quarantines (periods when contacts are restricted) should be added to the simulation (at regular intervals)
- "quarantine_len": integer specifying the number of days each quarantine should include
- "t_ei": integer indicating the time (in days) for an exposed person to become infectious
- "t_rs": integer indicating the time (in days) for a recovered person to become susceptible again
- "reinfection": percentage of nodes that can become susceptible again after recovering
- "comm_talk_freq": integer that specifies the average number of days needed for one node to contact all nodes that are labeled as friends
- "comm_talk_freq_change": integer indicating the step (in days) with which nodes reduce/increase contact rates with friends when there are/aren't restictions in effect
- "other_talk_freq": integer that specifies the average number of days needed for one node to contact all nodes that are labeled as acquaintances
- "other_talk_freq_change": integer indicating the step (in days) with which nodes reduce/increase contact rates with acquaintances when there are/aren't restictions in effect
- "p_er": probability for an exposed node to be asymptomatic, thus transitioning directly into recovered state (not becoming infectious)
- "gray_mass": probability of a node randomly becoming exposed, without a direct contact with an infectious node (usually a very low number)

As an addition, we made the simulations interactive, adding a slider for any parameter of choice that will be used to control given values for that parameter. This way we can easily identify how changing that particular parameter affects the outcome of the virus spread, while keeping all other parameters constant. 

### References:
- [LFR benchmark graph](https://en.wikipedia.org/wiki/Lancichinetti%E2%80%93Fortunato%E2%80%93Radicchi_benchmark)
- [Distribution of housholds in the US](https://www.statista.com/statistics/242189/disitribution-of-households-in-the-us-by-household-size/)
- [Proportion of asymptomatic cases](http://med.stanford.edu/content/dam/sm/id/documents/COVID/AsymptCOVID_TransmissionShip.pdf)
- [Length of immunity to reinfection after recovering](https://www.nejm.org/doi/full/10.1056/NEJMoa2034545)
- [Length of recovery period and modeling with negative binomial distribution](https://www.medrxiv.org/content/10.1101/2020.04.17.20069724v1.full.pdf?fbclid=IwAR0C_O-aZqGgcmmOnHkI3Eo_joKPsxjvClSgL5SvLFFNEhuNYzX-1Filvqo)
- [Length of incubation period](https://www.acpjournals.org/doi/full/10.7326/M20-0504)