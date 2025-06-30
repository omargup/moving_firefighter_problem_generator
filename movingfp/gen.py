import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import math 
import time
from itertools import combinations
from types import SimpleNamespace

import plotly.graph_objects as go
from movingfp.reds_utils import sum_costs, dist_toroidal


def init_fighter_pos(dim, fighter_coor, generator=None):
    if generator is None:
        generator = np.random.default_rng()

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
            fighter_pos.append(generator.uniform(0,1))

    return fighter_pos


def init_burnt_nodes(n, num_fires, generator=None):
    if generator is None:
        generator = np.random.default_rng()

    if n < 2:
        raise ValueError('The number `n` of nodes must be a integer greater or equal than 2.')
    
    if num_fires >= n or num_fires < 1 or type(num_fires) != int:
        raise ValueError('`num_fires` must be a integer number between 1 and n-1.')

    #burnt_nodes = random.sample(list(range(n)), k=num_fires)
    #burnt_nodes = generator.choice(list(range(n)), size=num_fires, replace=False)
    burnt_nodes = generator.permutation(n)[:num_fires]

    return burnt_nodes


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


def rand_distance_matrix(G, dist_interval, generator=None):
    if generator is None:
        generator = np.random.default_rng()

    if len(dist_interval) != 2:
        raise ValueError('Edge distance interval `dist_interval` must be a list of size 2; e.g.: [0.1, 3.5]')
    
    # Create Random Edge Weights
    min_dist, max_dist = dist_interval
    for (u, v) in G.edges():
        #dist = random.randint(min_dist, max_dist)
        dist = generator.uniform(min_dist, max_dist)
        G.edges[u, v]['weight'] = dist
    
    return nx.to_numpy_array(G)


def adjacency_matrix(G):
    """Computes the adjacency matrix of the graph G."""
    # Adjacency Matrix A in form of numpy array
    return nx.to_numpy_array(G).astype(int)


def complete_graph(n, dim, pos=None, generator=None):
    '''Builds a networkx complete graph.'''
    if generator is None:
        generator = np.random.default_rng()

    G = nx.complete_graph(n)
    if pos is None:
        pos = {v: [generator.random() for i in range(dim)] for v in range(n)}
    #TODO: check pos, dim and n
    nx.set_node_attributes(G, pos, "pos")
    
    return G


def erdos_graph(n, p, dim, generator=None):
    """Returns a networkx Erdos graph."""
    if generator is None:
        generator = np.random.default_rng()
        nx_seed = None
    else:
        nx_seed = int(generator.integers(low=0, high=2147483646))

    G = nx.erdos_renyi_graph(n, p, seed=nx_seed, directed=False)
    
    pos = {v: [generator.random() for i in range(dim)] for v in range(n)}
    nx.set_node_attributes(G, pos, "pos")
        
    return G


def erdos_graph_forced(n, p, dim,  generator=None):
    """Returns a connected erdos graph of size n.
    """
    #print(f'p: {p}')
    for current_n in range(n, n + int(n/4)):
        #print(f'n: {current_n}')
        for trial in range(20):  # attempts
            #print(f'trial: {trial}')
            G = erdos_graph(current_n, p, dim, generator)
            largest_cc = max(nx.connected_components(G), key=len)
            if len(largest_cc) == n:
                G = G.subgraph(largest_cc).copy()
                G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
                return G
    
    raise Exception("It was impossible to build a connected graph with the given parameters. Try again or tray with other parameters.")



def erdos_connected(n, p, dim=2, fighter_pos=None, num_fires=1, generator=None):
    """Returns a connected MFP Erdos instance in the unit cube of dimensions `dim`.
    
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
    num_fires : int, optional (default 1)
        Number of initial burnt nodes. `num_fires` nodes are chosen at random from `n` nodes.
    generator : None (default) or np.random._generator.Generator.
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

    if generator is None:
        generator = np.random.default_rng()
    if generator is not None and (type(generator) != np.random._generator.Generator):
        raise TypeError("´generator´ must be of type numpy.random._generator.Generator")
    
    #if seed is not None:
    #    ss = np.random.SeedSequence(seed)
    #    child_seeds = ss.spawn(3)
    #    rng = [np.random.default_rng(s) for s in child_seeds]

    # if seed is not None: generator = rng[0]
    fighter_pos = init_fighter_pos(dim, fighter_pos, generator)
    
    # if seed is not None: generator = rng[1] 
    burnt_nodes = init_burnt_nodes(n, num_fires, generator)

    # Fire graph
    # if seed is not None: generator = rng[2]
    G_fire = erdos_graph_forced(n, p, dim, generator)
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


def plot2d(inst):
    """ Draw a 2D-Moving Fire Fighter intance.
    Parameters
    ----------
    inst : a 2D Moving Fire Fighter intance."""

    G_fire = inst.G_fire
    G_fighter = inst.G_fighter
    agent_pos = inst.fighter_pos
    burnt_nodes = inst.burnt_nodes

    pos = nx.get_node_attributes(G_fire,'pos')
    if len(pos[0]) != 2:
        raise Exception("Only 2-dimensional graphs can be plotted with plot2d.")

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



def plot3d(inst, node_size=10, plot_labels=False, plot_grid=False):
    """ Draw a 3D-Moving Fire Fighter intance.
    Parameters
    ----------
    inst: a 3D Moving Fire Fighter intance.
    node_size: Int. The nodes' size.
    plot_axis: Plot axis and grid if True (Default False).
    plot_labels: Plot node labels if True (Default False)."""

    G_fire = inst.G_fire
    G_fighter = inst.G_fighter
    agent_pos = inst.fighter_pos
    burnt_nodes = list(inst.burnt_nodes)


    # Get node position in x, y and z.
    pos = nx.get_node_attributes(G_fire,'pos')
    if len(pos[0]) != 3:
        raise Exception("Only 3-dimensional graphs can be plotted with plot3d.")
    
    num_nodes = len(pos)

    pos_x = [pos[i][0] for i in range(num_nodes)] 
    pos_y = [pos[i][1] for i in range(num_nodes)]
    pos_z = [pos[i][2] for i in range(num_nodes)]


    # Get edge position in x, y, z. Each position includes
    # the starting and ending point.
    edge_list = G_fire.edges

    x_edges=[]
    y_edges=[]
    z_edges=[]

    for edge in edge_list:
        #format: [beginning, ending, None]
        x_coords = [pos[edge[0]][0], pos[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [pos[edge[0]][1], pos[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [pos[edge[0]][2], pos[edge[1]][2], None]
        z_edges += z_coords
    

    # Trace for edges
    trace_edges = go.Scatter3d(x=x_edges,
                            y=y_edges,
                            z=z_edges,
                            mode='lines',
                            showlegend= False,
                            line=dict(color='#073b4c',
                                        width=1.5),
                            opacity=0.8,
                            hoverinfo='none')
    
    #Trace for nodes
    node_labels = [i for i in range(num_nodes)]

    mode = 'markers'
    if plot_labels:
        mode = 'markers+text'

    trace_nodes = go.Scatter3d(x=pos_x,
                            y=pos_y,
                            z=pos_z,
                            mode=mode,
                            name='Undefended',
                            marker=dict(symbol='circle',
                                        size=node_size,
                                        color='#06d6a0', 
                                        opacity=0.8,
                                        line=dict(color='#06d6a0', width=0.5)),
                            text=node_labels,
                            hoverinfo='text')
    
    trace_fighter = go.Scatter3d(x=[agent_pos[0]],
                             y=[agent_pos[1]],
                             z=[agent_pos[2]],
                             mode='markers',
                             name='Anchor',
                             marker=dict(symbol='diamond',
                                         size=node_size,
                                         color='#118ab2',
                                         opacity=0.8,
                                         line=dict(color='#118ab2',
                                                   width=0.5)),
                             text = ['Anchor'],
                             hoverinfo = 'text')
    

    fires_x = [pos_x[i] for i in burnt_nodes]
    fires_y = [pos_y[i] for i in burnt_nodes]
    fires_z = [pos_z[i] for i in burnt_nodes]

    trace_fires = go.Scatter3d(x=fires_x,
                            y=fires_y,
                            z=fires_z,
                            mode='markers',
                            name='Initial fires',
                            marker=dict(symbol='circle',
                                            size=node_size,
                                            color='#ef476f',
                                            opacity=0.8,
                                            line=dict(color='#ef476f',
                                                    width=0.5)),
                                text = burnt_nodes,
                                hoverinfo = 'text')


    
    #we need to set the axis for the plot 

    grid_show = False
    if plot_grid:
        grid_show = True

    axis_x = dict(showbackground=False,
                
                showticklabels=grid_show ,
                zeroline=grid_show ,
                zerolinecolor="#6c757d",
                zerolinewidth = 0.5,

                showline = grid_show ,
                linecolor="#6c757d",
                linewidth = 0.5,
                
                showgrid=grid_show ,
                gridcolor="#6c757d",
                gridwidth=0.5,
                #griddash='dash',
                range = [-0.1,1.1],
                title='x' if plot_grid else '')

    axis_y = dict(showbackground=False,
                
                showticklabels=grid_show ,
                zeroline=grid_show ,
                zerolinecolor="#6c757d",
                zerolinewidth = 0.5,

                showline = grid_show ,
                linecolor="#6c757d",
                linewidth = 0.5,
                
                showgrid=grid_show ,
                gridcolor="#6c757d",
                gridwidth=0.5,
                #griddash='dash',
                range = [-0.1,1.1],
                title='y' if plot_grid else '')

    axis_z = dict(showbackground=False,
                
                showticklabels=grid_show ,
                zeroline=grid_show ,
                zerolinecolor="#6c757d",
                zerolinewidth = 0.5,

                showline = grid_show ,
                linecolor="#6c757d",
                linewidth = 0.5,
                
                showgrid=grid_show ,
                gridcolor="#6c757d",
                gridwidth=0.5,
                #griddash='5px',
                range = [-0.1,1.1],
                title='z' if plot_grid else '')
    


    # Layout for our plot
    layout = go.Layout(title="",
                width=750,
                height=650,
                showlegend=True,
                legend_title='',
                legend_yanchor='middle',
                
                
                legend_y= 0.5,
                scene=dict(xaxis=dict(axis_x),
                        yaxis=dict(axis_y),
                        zaxis=dict(axis_z)),
                margin=dict(l=80, r=80, b=80, t=80),
                hovermode='closest',
                
                
                )
    

    #Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes, trace_fighter, trace_fires]
    fig = go.Figure(data=data,
                    layout=layout)

    fig.show()