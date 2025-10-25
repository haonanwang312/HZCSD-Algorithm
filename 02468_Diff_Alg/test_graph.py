import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graph_generator import ring_graph, random_graph, metropolis_hastings,  MPIgraph, laplacian_matrix, mingyi_hong_adj

adj = random_graph(50, p = 0.4, seed = 2020)
# print(adj)
# L_minus, L_plus, degree_matrix = mingyi_hong_adj(adj)
# print(L_minus)
# print(L_plus)
# print(degree_matrix)

G = nx.Graph()
for i in range(len(adj)):
    for j in range(len(adj)):
        if adj[i, j]!= 0:  
          G.add_edge(i, j)

nx.draw(G, node_color = 'C0', node_size = 600, width = 2, with_labels = True)
plt.show()