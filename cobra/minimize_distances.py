import typer
import numpy as np
#%%
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
fig, ax = plt.subplots(figsize=(6,6))
ax.text(pos[0][0]-.1, pos[0][1]-.07, 'negatives',fontsize=18)
ax.text(pos[n_neg][0]-.2, 
    pos[n_neg][1]-.07, 'positives',fontsize=18)

#colors = nx.algorithms.bipartite.basic.color(G)
colors = np.concatenate([np.zeros(n_neg),np.ones(n_pos)])
nx.draw_networkx(
    G,
    pos = pos, 
    width = .1,
    arrows=False,
    node_color=['b' if color==0 else 'r' for color in colors],
    ax=ax)
fig.tight_layout()
#%%

print(colors)