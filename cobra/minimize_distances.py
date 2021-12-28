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
# %%
#Draw graph
num_negatives = 5
num_positives = 3
fig, ax = plt.subplots()
BG = nx.Graph()
source = ['s']
negatives = np.arange(num_negatives)
positives = np.arange(num_negatives, 
                (num_negatives+num_positives))
sink = ['t']

BG.add_nodes_from(source, bipartite=0)
BG.add_nodes_from(negatives, bipartite=1)
BG.add_nodes_from(positives, bipartite=2)
BG.add_nodes_from(sink, bipartite=3)
source_negatives_edges = []
negatives_positives_edges = []
positives_sink_edges = []
for neg in negatives:
    source_negatives_edges.append(('s', neg))
for neg in negatives:
    for pos in positives:
        negatives_positives_edges.append((neg, pos))
for pos in positives:
    positives_sink_edges.append((pos, 't'))

BG.add_edges_from(source_negatives_edges)
BG.add_edges_from(negatives_positives_edges)
BG.add_edges_from(positives_sink_edges)


nodes = BG.nodes()
# for each of the parts create a set 

nodes_0  = set([n for n in nodes if  BG.nodes[n]['bipartite']==0])
nodes_1  = set([n for n in nodes if  BG.nodes[n]['bipartite']==1])
nodes_2  = set([n for n in nodes if  BG.nodes[n]['bipartite']==2])
nodes_3  = set([n for n in nodes if  BG.nodes[n]['bipartite']==3])

# set the location of the nodes for each set
pos = dict()
pos.update( (n, (1, i+int(num_negatives/2))) for i, n in enumerate(nodes_0) ) # put nodes from X at x=1
pos.update( (n, (2, i)) for i, n in enumerate(nodes_1) ) # put nodes from Y at x=2
pos.update( (n, (3, i+1)) for i, n in enumerate(nodes_2) ) # put nodes from X at x=1
pos.update( (n, (4, i+int(num_negatives/2))) for i, n in enumerate(nodes_3) )

# set the colors
colors = ['tab:orange']
colors = colors + ['tab:blue' for i in range(num_negatives)]
colors = colors + ['tab:red' for i in range(num_positives)]
colors.append('tab:orange')
nx.draw_networkx(BG, pos=pos, ax=ax,
    node_color=colors,)
