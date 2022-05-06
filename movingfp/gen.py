import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import math 
import time
from itertools import combinations
from types import SimpleNamespace

from movingfp.reds_utils import sum_costs, dist_toroidal


def initial_config(n, dim, fighter_coor, num_burnt_nodes):
    if n < 2 or type(dim) != int:
        raise ValueError('The number `n` of nodes must be a integer greater or equal than 2.')
    
    if dim < 2 or type(dim) != int:
        raise ValueError('dimension `dim` must be a integer greater or equal than 2.')
    
    if fighter_coor is not None:
        if len(fighter_coor) != dim:
            raise ValueError('Fighter position must be a float list with the same dimension than dim.')
        fighter_pos = fighter_coor
        #TODO: should the fighter be restricted to being in the cube?
    else:
        fighter_pos = []
        for _ in range(dim):
            fighter_pos.append(random.uniform(0,1))
    
    if num_burnt_nodes >= n or num_burnt_nodes < 1 or type(num_burnt_nodes) != int:
        raise ValueError('`burnt_nodes` must be a integer number between 1 and n-1.')
    burnt_nodes = random.choices(list(range(n)), k=num_burnt_nodes)
    
    return fighter_pos, burnt_nodes


def euclidean_matrix(G):
    """Computes the euclidean distances matrix of the graph G."""
    nodes_pos = G.nodes(data="pos")
    for e in G.edges:
        u, v = e
        pos_u = nodes_pos[u]
        pos_v = nodes_pos[v]
        dist = math.dist(pos_u, pos_v)
        G.edges[u, v]['weight'] = dist
    # Distance Matrix D in form of numpy array
    return nx.to_numpy_array(G)


def rand_distance_matrix(G, dist_interval):
    if len(dist_interval) != 2:
        raise ValueError('Edge distance interval `dist_interval` must be a list of size 2; e.g.: [0.1, 3.5]')
    # Create Random Edge Weights
    min_dist, max_dist = dist_interval
    for (u, v) in G.edges():
        #dist = random.randint(min_dist, max_dist)
        dist = random.uniform(min_dist, max_dist)
        G.edges[u, v]['weight'] = dist
    return nx.to_numpy_array(G)


def adjacency_matrix(G):
    """Computes the adjacency matrix of the graph G."""
    # Adjacency Matrix A in form of numpy array
    return nx.to_numpy_array(G).astype(int)


def complete_graph(n, dim, pos=None):
    '''Builds a networkx complete graph.'''
    G = nx.complete_graph(n)
    if pos is None:
        pos = {v: [random.random() for i in range(dim)] for v in range(n)}
    #TODO: check pos, dim and n
    nx.set_node_attributes(G, pos, "pos")
    return G


def erdos_graph(n, p, dim, connected, seed):
    """Returns a networkx Erdos graph."""
    is_connected = False
    count = 0
    while not is_connected:
        ###########   
        G = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
        pos = {v: [random.random() for i in range(dim)] for v in range(n)}
        nx.set_node_attributes(G, pos, "pos")
        ###########
        count += 1
        if connected:
            is_connected = nx.is_connected(G)
            if not is_connected and count==20:
                raise Exception("After 20 attempts, it was impossible to build a connected graph with the given parameters. Try with other parameters.")
        else:
            is_connected = True
        
    return G


def geo_graph(n, r, dim, connected, seed):
    """Returns a networkx random geometric graph."""
    is_connected = False
    count = 0
    while not is_connected:
        ###########   
        G = nx.random_geometric_graph(n, r, dim=dim, p=2, seed=seed)
        ###########
        count += 1
        if connected:
            is_connected = nx.is_connected(G)
            if not is_connected and count==20:
                raise Exception("After 20 attempts, it was impossible to build a connected graph with the given parameters. Try with other parameters.")
        else:
            is_connected = True
    return G


def spanning_tree(G):
    """Returns a networkx spanning tree or forest on an undirected graph G."""
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.uniform(0, 1)
        T = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal', ignore_nan=False)
    for (u, v, attr) in G.edges(data=True):
        attr.clear()
    for (u, v, attr) in T.edges(data=True):
        attr.clear()
    return T, G


def bfs_tree(G):
    """Returns a networkx bfs tree on an undirected graph G."""
    T = nx.create_empty_copy(G, with_data=True)
    bfs_edges = nx.bfs_edges(G, source=random.randint(0, G.number_of_nodes()-1), reverse=False, depth_limit=None, sort_neighbors=None)
    T.add_edges_from(bfs_edges)
    return T, G


def reds_graph(n, R, E, S, t):
    """
    Code from: https://github.com/jesgadiaz/REDS_creator
    """
    G = nx.Graph()
    nodes = []
    for i in range(n):
        x = random.random()
        y = random.random()
        p = (x,y)
        nodes.append(p)
        G.add_node(i, pos=(x,y))
    d = []
    for i in range(n):
        temp = [0 for p in range(n)]
        d.append(temp)
    for i in range(n):
        for j in range(n):
            #d[i][j] = dist(nodes[i],nodes[j])
            d[i][j] = dist_toroidal(nodes, i,j)
    NR = []
    for i in range(n):
        temp = []
        for j in range(n):
            if i != j and d[i][j] < R:
                temp.append(j)
        NR.append(temp)
    start_time = time.time()
    while(time.time() - start_time < t):
        i = random.randint(0,n-1)
        N = [n for n in G[i]]
        N2 = []
        for j in NR[i]:
            if j not in N:
                N2.append(j)
        if len(N2) != 0:
            j = N2[random.randint(0,len(N2)-1)]
            G.add_edge(i,j)
            if not (E >= sum_costs(G,d,i,j,S) and E >= sum_costs(G,d,j,i,S)):
                G.remove_edge(i,j)
    return G


def erdos(n, p, dim=2, fighter_pos=None, burnt_nodes=1, connected= True, seed=None):
    """Returns a WFFP Erdos instance in the unit cube of dimensions `dim`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability for edge creation.
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_fire = erdos_graph(n, p, dim, connected, seed)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = euclidean_matrix(G_fighter)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance


def geo(n, r, dim=2, fighter_pos=None, burnt_nodes=1, connected= True, seed=None):
    """Returns a WFFP Random Geometric instance in the unit cube of dimensions `dim`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    r : float
        The distance threshold. Edges are included in the edge list if the distance between the two nodes is less than `r`.
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_fire = geo_graph(n, r, dim, connected, seed)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = euclidean_matrix(G_fighter)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance


def nm_erdos(n, p, dist_interval, dim=2, fighter_pos=None, burnt_nodes=1, connected= True, seed=None):
    """Returns a WFFP No Metric Erdos instance in the unit cube of dimensions `dim`, with distances in the interval `dist_interval`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability for edge creation.
    dist_interval : list
        Edges distances are sampled at random from the interval `dist_interval`.
        i.e: [min_val, max_val].
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_fire = erdos_graph(n, p, dim, connected, seed)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = rand_distance_matrix(G_fighter, dist_interval)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance


def reds(n, R, E, S, t, dim=2, fighter_pos=None, burnt_nodes=1, seed=None):
    """Returns a WFFP REDS instance in the unit cube of dimensions `dim`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    R : float [0, 1]
        Reach.
    E : float [0, 1]
        Energy.
    S : float [0, 1]
        Sinergy.
    t : int
        maximum time in seconds (stop condition).
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes
    
    References
    ---------
    Code: https://github.com/jesgadiaz/REDS_creator
    Paper: https://eprints.soton.ac.uk/364826/
    """


    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_fire = reds_graph(n, R, E, S, t)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = euclidean_matrix(G_fighter)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance






def erdos_stree(n, p, dim=2, fighter_pos=None, burnt_nodes=1, connected= True, seed=None):
    """Returns a WFFP Erdos spanning tree (or forest) instance in the unit cube of dimensions `dim`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability for edge creation.
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_ = erdos_graph(n, p, dim, connected, seed)
    G_fire, _ = spanning_tree(G_)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = euclidean_matrix(G_fighter)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance


def geo_stree(n, r, dim=2, fighter_pos=None, burnt_nodes=1, connected= True, seed=None):
    """Returns a WFFP Random Geometric spanning tree (or forest) instance in the unit cube of dimensions `dim`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    r : float
        The distance threshold. Edges are included in the edge list if the distance between the two nodes is less than `r`.
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_ = geo_graph(n, r, dim, connected, seed)
    G_fire, _ = spanning_tree(G_)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = euclidean_matrix(G_fighter)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance

def nm_erdos_stree(n, p, dist_interval, dim=2, fighter_pos=None, burnt_nodes=1, connected= True, seed=None):
    """Returns a WFFP No Metric Erdos spanning tree (or forest) instance in the unit cube of dimensions `dim`,
    with distances in the interval `dist_interval`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability for edge creation.
    dist_interval : list
        Edges distances are sampled at random from the interval `dist_interval`.
        i.e: [min_val, max_val].
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_ = erdos_graph(n, p, dim, connected, seed)
    G_fire, _ = spanning_tree(G_)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = rand_distance_matrix(G_fighter, dist_interval)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance


def reds_stree(n, R, E, S, t, dim=2, fighter_pos=None, burnt_nodes=1, seed=None):
    """Returns a WFFP REDS spanning tree (or forest) instance in the unit cube of dimensions `dim`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    R : float [0, 1]
        Reach.
    E : float [0, 1]
        Energy.
    S : float [0, 1]
        Sinergy.
    t : int
        maximum time in seconds (stop condition).
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes
    
    References
    ---------
    Code: https://github.com/jesgadiaz/REDS_creator"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_ = reds_graph(n, R, E, S, t)
    G_fire, _ = spanning_tree(G_)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = euclidean_matrix(G_fighter)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance


def erdos_bfstree(n, p, dim=2, fighter_pos=None, burnt_nodes=1, connected= True, seed=None):
    """Returns a WFFP Erdos BFS tree instance in the unit cube of dimensions `dim`.
    
    The starting node for breadth-first search is chosen at random from V. BFS iterates over only
    those edges in the component reachable from this node.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability for edge creation.
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_ = erdos_graph(n, p, dim, connected, seed)
    G_fire, _ = bfs_tree(G_)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = euclidean_matrix(G_fighter)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance


def geo_bfstree(n, r, dim=2, fighter_pos=None, burnt_nodes=1, connected= True, seed=None):
    """Returns a WFFP Random Geometric BFS Tree instance in the unit cube of dimensions `dim`.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    r : float
        The distance threshold. Edges are included in the edge list if the distance between the two nodes is less than `r`.
    dim : int, optional (default 2)
        Dimension of graph.
    fighter_pos : list or None (default)
        The Firefighter initial position. If `None`, position is chosen at random in the unit cube
        of dimensions `dim`.
    burnt_nodes : int, optional (default 1)
        Number of initial burnt nodes. `burnt_nodes` nodes are chosen at random from `n` nodes.
    seed : int or None (default)
        Indicator of random number generation state.
    
    Returns
    -------
    A_fire : numpy array
        n x n adjacency matrix of the fire graph.
    D_fighter : numpy array
        (n+1) x (n+1) distance matrix of the firefighter graph. The last row and column corresponds
        to the firefighter.
    burnt_nodes : list
        Initial burnt nodes.
    fighter_pos : list
        Initial firefighter position.
    node_pos : list
        Positions of the nodes"""

    random.seed(seed)
    fighter_pos, burnt_nodes = initial_config(n, dim, fighter_pos, burnt_nodes)

    # Fire graph
    G_ = geo_graph(n, r, dim, connected, seed)
    G_fire, _ = bfs_tree(G_)
    A_fire = adjacency_matrix(G_fire)

    # Firefighter graph
    pos = nx.get_node_attributes(G_fire,'pos')
    pos[n] = fighter_pos
    G_fighter = complete_graph(n+1, dim, pos)
    D_fighter = euclidean_matrix(G_fighter)

    nodes_pos = [coor for u, coor in pos.items()][:-1]
    instance = {'A': A_fire, 'D': D_fighter, 'fighter_pos': fighter_pos,
        'node_pos': nodes_pos, 'burnt_nodes': burnt_nodes, 'G_fire': G_fire,
        'G_fighter': G_fighter}
    instance = SimpleNamespace(**instance)
    return instance


def draw_ffp(inst):
    """ Draw a 2D-Walking Fire Fighter intance.
    Parameters
    ----------
    inst : a Walking Fire Fighter intance."""

    G_fire = inst.G_fire
    G_fighter = inst.G_fighter
    agent_pos = inst.fighter_pos
    burnt_nodes = inst.burnt_nodes

    pos = nx.get_node_attributes(G_fire,'pos')
    if len(pos[0]) != 2:
        raise Exception("Only 2-dimensional graphs can be plotted")

    fig, ax = plt.subplots(figsize=(7,7))

    # G_fighter
    pos = nx.get_node_attributes(G_fighter,'pos')
    nx.draw_networkx_nodes(G_fighter, pos, ax=ax, node_size=150, node_color='#ABB2B9', alpha=None)
    nx.draw_networkx_edges(G_fighter, pos, width=1.0, edge_color='#ABB2B9', alpha=None, ax=None,  label=None, node_size=150, style='dashed')

    # G_fire
    pos = nx.get_node_attributes(G_fire,'pos')
    nx.draw_networkx_nodes(G_fire, pos, ax=ax, node_size=150, node_color='#48C9B0', alpha=None)
    nx.draw_networkx_edges(G_fire, pos, width=1.0, edge_color='#48C9B0', alpha=None, ax=None,  label=None, node_size=150)

    # Fighter
    G_agent = nx.Graph()
    G_agent.add_node(nx.number_of_nodes(G_fire), pos=agent_pos)
    a_pos = nx.get_node_attributes(G_agent,'pos')
    nx.draw_networkx_nodes(G_agent, pos=a_pos, ax=ax, node_size=150, node_color='#3498DB', alpha=None, node_shape='D')

    # Burnt
    G_burnt = nx.Graph()
    pos = nx.get_node_attributes(G_fire,'pos')
    for node in burnt_nodes:
        G_burnt.add_node(node, pos=pos[node])
    b_pos = nx.get_node_attributes(G_burnt,'pos')
    nx.draw_networkx_nodes(G_burnt, pos=b_pos, ax=ax, node_size=150, node_color='#E74C3C', alpha=None)

    # Node labels
    pos = nx.get_node_attributes(G_fighter,'pos')
    labels={n: n for n in G_fighter}
    nx.draw_networkx_labels(G_fighter, pos, labels=labels, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, clip_on=True)
    
    #if edge_labels == True:
    #    pos = nx.get_node_attributes(G_fighter,'pos')
    #    dist = nx.get_edge_attributes(G_fighter,'weight')
    #    nx.draw_networkx_edge_labels(G_fighter, pos=pos, ax=ax, edge_labels=dist, font_color='k', alpha=None, font_size=8)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    limits = [ -0.1, 1.1, -0.1, 1.1]
    plt.axis(limits)
    plt.axis('on')
    plt.show()