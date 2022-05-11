import numpy as np

import random
import yaml
import os
from types import SimpleNamespace

import movingfp.gen as mfp

from typing import List, Union


def mfp2yaml(x, file_path, file_name):
    x_dict = {'A': x.A.tolist(),
              'D': x.D.tolist(),
              'fighter_pos': x.fighter_pos,
              'node_pos': x.node_pos,
              'burnt_nodes': x.burnt_nodes}

    with open(os.path.join(file_path, file_name), 'w') as file:
        yaml.dump(x_dict, file, default_flow_style=False)


def yaml2dict(file_name):
    with open(file_name, 'r') as f:
        try:
            x = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return x


def dict2mfp(x):
    x['A'] = np.array(x['A'])
    x['D'] = np.array(x['D'])
    instance = SimpleNamespace(**x)
    return instance

def yaml2mfp(file_name):
    return dict2mfp(yaml2dict(file_name))


def erdos_data(n_list: List[int] = [8],
               p_list: Union[List[float], None] = [0.5],
               pn_list: Union[List[float], None] = None,
               dim_list: List[int] = [2],
               bn_list: List[int] = [1],
               num_graphs: int = 5,
               connected: bool = False,
               file_path: Union[str, None] = None,
               seed: Union[int, None] = None):
    """
    Example
    -------
    5 Erdos MFP instances with edge probability 0.5 and 0.7 for 10, 20 and 30 nodes:
    >>> erdos_data(n_list=[10,20,30], p_list=[0.5, 0.7], num_graphs=5)
    
    5 Erdos MFP instances with edge probability 2.5/n for n=10, 20 and 30:
    >>> erdos_data(n_list=[10,20,30], pn_list=[2.5], num_graphs=5)

    
    """
    if file_path is None:
        file_path = 'data/erdos'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    if p_list is None and pn_list is None:
        raise ValueError("Only one between `p_list` and `pn_list` can be `None`")
        
    proportional = False
    if pn_list is not None:
        proportional = True


    for n in n_list:
        
        if proportional:
            p_list = []
            for p in pn_list: 
                p_list.append(p/n)
        
        for p in p_list:
            for dim in dim_list:
                for burnt_nodes in bn_list:
                    for num in range(num_graphs):
                        seed = None
                        #TODO: seed to generate random seeds
                        x = mfp.erdos(n, p, dim, None, burnt_nodes, connected, seed)
                        file_name = 'erdos_n' + str(n) \
                                    + '_p' + str(p).replace('.', '') \
                                    + '_d' + str(dim) \
                                    + '_b' + str(burnt_nodes) \
                                    + '_c' + ('t' if connected else 'f') \
                                    + '_id' + str(num) + '.yaml'
                        mfp2yaml(x, file_path, file_name)