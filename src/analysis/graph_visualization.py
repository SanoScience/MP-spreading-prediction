import networkx as nx
from networkx.classes.function import get_edge_attributes
import numpy as np
import matplotlib.pyplot as plt

def build_graph(matrix, features):
    G = nx.from_numpy_matrix(matrix)
    features_dict = dict(zip(range(matrix.shape[0]), features))
    nx.set_node_attributes(G, features_dict, "concentration")
    return G

def draw_graph(G):
    pos = nx.circular_layout(G)
    edges_attr = get_edge_attributes(G, 'weight')

    nx.draw(G, node_size=5)
    plt.show()
    

if __name__ == '__main__':
    adj_matrix_path = '../../data/connectomes/sub-AD4009/connect_matrix_rough.csv'
    concentration_path = '../../data/PET_regions_concentrations_t0/sub-AD4009/sub-AD4009_ses-1_acq-AP_pet-abeta-av45[2011-07-07]_opt.csv'
    adj_matrix = np.array(np.genfromtxt(adj_matrix_path, delimiter=','))
    concentrations = np.array(np.genfromtxt(concentration_path, delimiter=','))
    
    G = build_graph(adj_matrix, concentrations)
    draw_graph(G)
    