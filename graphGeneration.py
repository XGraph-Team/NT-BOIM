import networkx as nx
import torch_geometric.datasets as ds
import random
import ndlib
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

from torch_geometric.datasets import Planetoid

def connSW(n):
    g = nx.connected_watts_strogatz_graph(n, int(n/200), 0.1)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config

def BA(n):
    g = nx.barabasi_albert_graph(n, 10)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config

def ER(n):

    g = nx.erdos_renyi_graph(n, 20/n)

    while nx.is_connected(g) == False:
        g = nx.erdos_renyi_graph(n, 0.05)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40, 80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def CiteSeer():
    dataset = Planetoid(root='./Planetoid', name='CiteSeer')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)
    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def PubMed():
    dataset = Planetoid(root='./Planetoid', name='PubMed')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)
    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def Cora():
    dataset = Planetoid(root='./Planetoid', name='Cora')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    # data.num_classes = dataset.num_classes
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)
    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight
    # return g, config, data
    return g, config


def photo():

    dataset = ds.Amazon(root='./geo', name = 'Photo')
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)
    g = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def coms():

    dataset = ds.Amazon(root='./geo', name = 'Computers')
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)
    g = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def synn():

    edgelist = []
    for i in range(3):
        for j in range(i+1,3):
            edgelist.append((i,j))

    for i in range(3,21):
        edgelist.append((0,i))
        edgelist.append((1,i+18))
        edgelist.append((2,i+36))

    for i in range(18):
        edgelist.append((i+3,i+57))
        edgelist.append((i+21,i+75))
        edgelist.append((i+39,i+93))

    for i in range(5):
        edgelist.append((94, 95+i))
        edgelist.append((95+i, 100+i))

        edgelist.append((105, 106+i))
        edgelist.append((106+i, 111+i))

        edgelist.append((116, 117+i))
        edgelist.append((117+i, 122+i))

        edgelist.append((127, 128+i))
        edgelist.append((128+i, 133+i))

    edgelist.append((57,100))
    edgelist.append((58,111))
    edgelist.append((75,122))
    edgelist.append((93,133))

    g=nx.from_edgelist(edgelist)
    
    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def synn_small():
    edgelist = []
    # Reduced nodes in each range
    for i in range(2):  # Reduced from 3
        for j in range(i+1, 2):  # Reduced from 3
            edgelist.append((i, j))

    # Smaller outer loops
    for i in range(2, 7):  # Reduced from 21, smaller increments
        edgelist.append((0, i))
        edgelist.append((1, i + 5))  # Adjusted the offsets

    # Further reduced nested loops
    for i in range(5):  # Reduced drastically from 18
        edgelist.append((i+2, i + 7))  # Adjusted indices
        edgelist.append((i+7, i + 12))  # Adjusted further

    # Very simplified connections at the end
    edgelist.append((7, 12))
    edgelist.append((8, 13))

    g = nx.from_edgelist(edgelist)
    config = mc.Configuration()

    # Adding weights as in the original function
    for a, b in g.edges():
        weight = random.randrange(40, 80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def power_law(n, m=2):
    """
    Generate a power law graph using the Barabási-Albert model.
    
    :param n: Number of nodes
    :param m: Number of edges to attach from a new node to existing nodes
    :return: NetworkX graph and Configuration object
    """
    g = nx.barabasi_albert_graph(n, m)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40, 80)
        weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config


def bipartite_graph(n1, n2):
    """
    Generate a bipartite graph.
    
    :param n1: Number of nodes in set 1
    :param n2: Number of nodes in set 2
    :return: NetworkX graph and Configuration object
    """
    g = nx.bipartite.random_graph(n1, n2, 0.1)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40, 80)
        weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config

def highly_clustered_graph(n, p):
    """
    Generate a highly clustered graph using the Watts-Strogatz model.
    
    :param n: Number of nodes
    :param p: Rewiring probability
    :return: NetworkX graph and Configuration object
    """
    g = nx.watts_strogatz_graph(n, k=int(n/10), p=p)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40, 80)
        weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config

def disconnected_graph(n, m, num_components):
    """
    Generate a disconnected graph composed of multiple components.
    
    :param n: Total number of nodes
    :param m: Number of edges per component
    :param num_components: Number of components
    :return: NetworkX graph and Configuration object
    """
    components = [nx.barabasi_albert_graph(int(n/num_components), m) for _ in range(num_components)]
    g = nx.disjoint_union_all(components)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40, 80)
        weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config

def star_graph(n):
    """
    Generate a star graph.
    
    :param n: Number of nodes
    :return: NetworkX graph and Configuration object
    """
    g = nx.star_graph(n-1)  # nx.star_graph creates a star graph with n nodes (0 to n-1, with 0 as the center)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40, 80)
        weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config
