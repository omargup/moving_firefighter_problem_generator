# 📦 Moving/Walking  Firefighter problem instances generator

This package build on top of Networkx contains instances generators of the Moving/Walking Firefighter Problem, proposed by the Network and Data Science Laboratory CIC-IPN México.

- **Source:** https://github.com/...
- **Tutorial:** https://...
- **Website:** https://...

Currently, these generators are available:

* Erdos
* Geometric
* No Metric Erdos
* REDS
* Spanning Trees
  * Erdos Spanning Tree
  * Geometric Spanning Tree
  * No Metric Erdos Spanning Tree
  * REDS Spanning Tree
* BFS Trees
  * Erdos BFS Tree
  * Geometric BFS Tree
  * No Metric Erdos BFS Tree
  * REDS BFS Tree



## Install

Install the last version of movingfp

```bash
pip install movingfp
```

## Simple example

Create an erdos instance and access its attributes

```bash
>>> import movingfp.gen as mfp
>>> x = mfp.erdos(n=8, p=0.5)
>>> x.A
>>> x.D
>>> x.fighter_pos
>>> x.node_pos
>>> x.burnt_nodes
>>> G_fire = x.G_fire
>>> G_fighter = x.G_fighter
```

Draw the instance

```bash
>>> mfp.draw_ffp(x)
```

![Erdos instance](img/erdos_instance.png)

## To Do

## License
