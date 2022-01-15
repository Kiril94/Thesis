#%%
import typer
import numpy as np
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#%%
n_neg = 10
n_pos = 3
#graph_arr = np.array([np.repeat(np.arange(n_neg),n_pos),
#    np.tile(np.arange(n_pos), n_neg)]).T
neg_neg_arr = np.zeros((n_neg, n_neg))
pos_pos_arr = np.zeros((n_pos, n_pos))
pos_neg_arr = np.ones((n_pos, n_neg))
neg_pos_arr = pos_neg_arr.T
adj_matrix_top = np.concatenate([neg_neg_arr, neg_pos_arr], axis=1)
adj_matrix_bot = np.concatenate([pos_neg_arr, pos_pos_arr], axis=1)

adj_matrix = np.concatenate([adj_matrix_top,adj_matrix_bot], 
    axis=0)
G = nx.DiGraph(adj_matrix)
pos = nx.drawing.layout.bipartite_layout(G, nodes=np.arange(n_neg))


#%%
G = nx.DiGraph()

G.add_edges_from(
    [
        (1, 2, {"capacity": 1, "weight": 0}),
        (1, 3, {"capacity": 1, "weight": 0}),
        (1, 4, {"capacity": 1, "weight": 0}),
        (1, 5, {"capacity": 1, "weight": 0}),
        (1, 6, {"capacity": 1, "weight": 0}),
        (1, 7, {"capacity": 1, "weight": 0}),
        (1, 8, {"capacity": 1, "weight": 0}),
        (1, 9, {"capacity": 1, "weight": 0}),
        (1, 10, {"capacity": 1, "weight": 0}),

        (2, 11, {"capacity": 1, "weight": 9}),
        (2, 12, {"capacity": 1, "weight": 10}),
        (2, 13, {"capacity": 1, "weight": 1}),
        (3, 11, {"capacity": 1, "weight": 10}),
        (3, 12, {"capacity": 1, "weight": 20}),
        (3, 13, {"capacity": 1, "weight": 11}),
        (4, 11, {"capacity": 1, "weight": 30}),
        (4, 12, {"capacity": 1, "weight": 22}),
        (4, 13, {"capacity": 1, "weight": 40}),
        (5, 11, {"capacity": 1, "weight": 21}),
        (5, 12, {"capacity": 1, "weight": 30}),
        (5, 13, {"capacity": 1, "weight": 2}),
        (6, 11, {"capacity": 1, "weight": 11}),
        (6, 12, {"capacity": 1, "weight": 40}),
        (6, 13, {"capacity": 1, "weight": 3}),
        (7, 11, {"capacity": 1, "weight": 2}),
        (7, 12, {"capacity": 1, "weight": 30}),
        (7, 13, {"capacity": 1, "weight": 10}),
        (8, 11, {"capacity": 1, "weight": 15}),
        (8, 12, {"capacity": 1, "weight": 21}),
        (8, 13, {"capacity": 1, "weight": 30}),
        (9, 11, {"capacity": 1, "weight": 30}),
        (9, 12, {"capacity": 1, "weight": 26}),
        (9, 13, {"capacity": 1, "weight": 10}),
        (10, 11, {"capacity": 1, "weight": 40}),
        (10, 12, {"capacity": 1, "weight": 23}),
        (10, 13, {"capacity": 1, "weight": 25}),


        (11, 14, {"capacity": 3, "weight": 0}),
        (12, 14, {"capacity": 3, "weight": 0}),
        (13, 14, {"capacity": 3, "weight": 0}),
    ]
)

mincostFlow = nx.max_flow_min_cost(G, 1, 14)
mincost = nx.cost_of_flow(G, mincostFlow)
print(mincost)
print(mincostFlow)
#%%
G = nx.DiGraph()

G.add_edges_from(
    [
        (1, 2, {"capacity": 1, "weight": 0}),
        (1, 3, {"capacity": 1, "weight": 0}),
        (1, 4, {"capacity": 1, "weight": 0}),
        (1, 5, {"capacity": 1, "weight": 0}),
        (1, 6, {"capacity": 1, "weight": 0}),

        (2, 7, {"capacity": 1, "weight": 9}),
        (2, 8, {"capacity": 1, "weight": 10}),
        (2, 8, {"capacity": 1, "weight": 10}),
        (3, 7, {"capacity": 1, "weight": 10}),
        (3, 8, {"capacity": 1, "weight": 20}),
        (4, 7, {"capacity": 1, "weight": 30}),
        (4, 8, {"capacity": 1, "weight": 22}),
        (5, 7, {"capacity": 1, "weight": 21}),
        (5, 8, {"capacity": 1, "weight": 30}),
        (6, 7, {"capacity": 1, "weight": 11}),
        (6, 8, {"capacity": 1, "weight": 40}),
        
        (7, 9, {"capacity": 2, "weight": 0}),
        (8, 9, {"capacity": 2, "weight": 0}),
    ]
)

mincostFlow = nx.max_flow_min_cost(G, 1, 9)
mincost = nx.cost_of_flow(G, mincostFlow)
print(mincost)
print(mincostFlow)