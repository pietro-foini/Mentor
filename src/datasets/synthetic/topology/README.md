# Topology

## Overview

Let's create several directed subgraphs/teams by populating them with three different possible patterns of connections (motifs):

<p align="center">
  <img src="motifs.png" width="600">
</p>

If a team is created using a particular motif, this means that the connectivity pattern between its own members is 
generated by repeatedly adding one of the three predefined motifs (delivering a 3-class problem).

### TV1

These sub-graphs are inserted into a bigger directed graph ruled by Erdos Renyi (ER) model. The hyperparameters of the 
underlying connectivity defined by the ER model allow us to control the amount of global noise by adding random edges 
not relevant to the classification problem. In this case, we used a medium-high noise effect.

### TV2

These sub-graphs are inserted into a bigger directed graph ruled by Erdos Renyi (ER) model. The hyperparameters of the 
underlying connectivity defined by the ER model allow us to control the amount of global noise by adding random edges 
not relevant to the classification problem. In this case, we used a low noise effect. Furthermore, we add new edges 
into the network following a preferential attachment method: a member of each team obtains an amount of edges from 
other nodes of other teams (similar to algorithm of *in-degree* synthetic dataset). 

### TV3

These sub-graphs are inserted into a bigger directed graph ruled by Erdos Renyi (ER) model. The hyperparameters of the 
underlying connectivity defined by the ER model allow us to control the amount of global noise by adding random edges 
not relevant to the classification problem. In this case, we used a medium-high noise effect. 
This implementation have nodes that can belong to multiple teams and nodes that not belong to any team.

## Data

The *data* folder contains sub-folders which contains:

- ```graph.pkl```: this file must contain the Networkx graph you want to work with.
- ```teams_label.pkl```: this file is a dictionary where the keys are team_id and the values are the corresponding classes.

