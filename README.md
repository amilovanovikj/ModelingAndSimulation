# ModelingAndSimulation
Simulation of COVID spread in generated network using a SEIR model

- Work with SEIR model:
    - Suspectible: can get exposed
    - Exposed: can expose neighbors, transitions to infected after X number of days (tested positive), or stay asymptomatic and directly transition to recovered after X number of days
    - Infected: is isolated and cannot infect neighbors, transitions to recovered after X number of days
    - Recovered: cannot get exposed again
- Exposed nodes can expose neighbors with given probability X (define value from WHO)
- Gray mass: any suspectible node can get exposed from gray mass with given (very low) probability X
- Start with initial percentage of population as exposed
- Define family: a group of nodes that have contact with each other every day
- All other neighbors of a node that are not family have contact with that node after X days
- Define asymptomatic probability of an exposed node (might not be used)
- Find ideal way of generating the graph, in order to represent a network of contacts as realistically as possible:
    - Nodes that are family members create a complete subgraph
    - After generating the families using a distribution of family sizes, connect the graph with edges that represent friends
- Distribute contacts with friends (not all nodes will have contact with their friends on the same day)
- Calculate R0 for all simulations

PARAMETERS:
    - s_to_e: probability to transition from suspectible to exposed after having contact
    - e_to_i: time (in days) for an exposed node to transition to infected
    - i_to_r: time (in days) for an infected node to transition to recovered
    - e_to_r: time (in days) for an exposed node to transition to recovered (might not be used)
    - gray_mass: probability of becoming exposed from gray mass
    - asympt: probability of staying asymptomatic (might not be used)
    - init_ex: percentage of initialy exposed nodes
    - friend_contact: time (in days) after which nodes meet with their friends -> this is a hyperparameter

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Next steps:
- add distribution to incubation and recovery period
- write function to display graph characteristics
- add different contact intervals for family members, friends and acquaintances
- implement the functionality that infected nodes can expose family members
- OPTIONAL: make the model SEIRS and add distribution to losing immunity
- OPTIONAL: make dynamic changes to contact intervals to reflect government measures
