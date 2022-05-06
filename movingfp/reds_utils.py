import math 

def sum_costs(G, d, i,j,S):
    """
    Code from: https://github.com/jesgadiaz/REDS_creator
    """
    Ni = [n for n in G[i]]
    Nj = [n for n in G[j]]
    set_Ni = set(Ni)
    set_Nj = set(Nj)
    k = len(set_Ni.intersection(set_Nj))
    c = 0
    for p in Ni:
        c = (d[i][j] / (1 + S * k)) + c
    return c

def dist_toroidal(nodes, i,j):
    """
    Code from: https://github.com/jesgadiaz/REDS_creator
    """
    p1 = nodes[i]
    p2 = nodes[j]
    dx = 0
    dy = 0
    if p1[0] <= p2[0]:
        dx = min(p1[0] + (1-p2[0]), p2[0] - p1[0])
    else:
        dx = min(p2[0] + (1-p1[0]), p1[0] - p2[0])
    if p1[1] <= p2[1]:
        dy = min(p1[1] + (1-p2[1]), p2[1] - p1[1])
    else:
        dy = min(p2[1] + (1-p1[1]), p1[1] - p2[1])
    d = math.sqrt(dx**2 + dy**2)
    return d