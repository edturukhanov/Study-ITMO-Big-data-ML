# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:55:13 2019

@author: fonma
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
%matplotlib qt


graph = nx.gnm_random_graph(30,300, directed=False)

print(nx.info(graph))
adj_matr = nx.to_numpy_matrix(graph)
print(adj_matr)
nx.draw(graph, pos = nx.spectral_layout(graph), 
        node_color='blue', node_size=40, with_labels=False)
plt.imshow(adj_matr)
plt.title('Visualization of adjacency matrix')
plt.show()

for (u, v) in graph.edges():
    graph.edges[u,v]['weight'] = random.randint(0,10)

adj_matr = nx.to_numpy_matrix(graph)
print(adj_matr)

plt.imshow(adj_matr)
plt.title('Visualization of adjacency matrix')
plt.show()
############## Solution

graph = nx.gnm_random_graph(100,2000, directed=False)

for (u, v) in graph.edges():
    graph.edges[u,v]['weight'] = random.randint(0,10)

pos = nx.spring_layout(graph)
adj_matr = nx.to_numpy_matrix(graph)
print(adj_matr)
nx.draw(graph, pos = pos, 
        node_color='blue', node_size=200, with_labels=True)

### Dijakstra
import timeit

total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for i in range(0,10,1):
    t = timeit.default_timer()
    nx.single_source_dijkstra(graph, 1)
    elapsed_time = timeit.default_timer() - t
    total_time.append(elapsed_time)
vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)

vector_mean.Time
0.00025633
0.00033027
0.0003085

### Bellman Ford
total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for i in range(0,10,1):
    t = timeit.default_timer()
    nx.single_source_bellman_ford(graph, 1)
    elapsed_time = timeit.default_timer() - t
    total_time.append(elapsed_time)
vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)






from collections import deque
from networkx.utils import generate_unique_node

def negative_edge_cycle(G, weight='weight'):
    """
    If there is a negative edge cycle anywhere in G, returns True.
    Also returns the total weight of the cycle and the nodes in the cycle.
    Parameters
    ----------
    G : NetworkX graph
    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight
    Returns
    -------
    length : numeric
        Length of a negative edge cycle if one exists, otherwise None.
    nodes: list
        Nodes in a negative edge cycle (in order) if one exists,
        otherwise None.
    negative_cycle : bool
        True if a negative edge cycle exists, otherwise False.
    Examples
    --------
    >>> import networkx as nx
    >>> import bellmanford as bf
    >>> G = nx.cycle_graph(5, create_using = nx.DiGraph())
    >>> print(bf.negative_edge_cycle(G))
    (None, [], False)
    >>> G[1][2]['weight'] = -7
    >>> print(bf.negative_edge_cycle(G))
    (-3, [4, 0, 1, 2, 3, 4], True)
    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.
    This algorithm uses bellman_ford() but finds negative cycles
    on any component by first adding a new node connected to
    every node, and starting bellman_ford on that node.  It then
    removes that extra node.
    """
    newnode = generate_unique_node()
    G.add_edges_from([(newnode, n) for n in G])

    try:
        pred, dist, negative_cycle_end = bellman_ford_tree(G, newnode, weight)

        if negative_cycle_end:
            nodes = []
            negative_cycle = True
            end = negative_cycle_end
            while True:
                nodes.insert(0, end)
                if nodes.count(end) > 1:
                    end_index = nodes[1:].index(end) + 2
                    nodes = nodes[:end_index]
                    break
                end = pred[end]
            length = sum(
                G[u][v].get(weight, 1) for (u, v) in zip(nodes, nodes[1:])
            )
        else:
            nodes = None
            negative_cycle = False
            length = None

        return length, nodes, negative_cycle
    finally:
        G.remove_node(newnode)


def bellman_ford(G, source, target, weight='weight'):
    """
    Compute shortest path and shortest path lengths between a source node
    and target node in weighted graphs using the Bellman-Ford algorithm.
    Parameters
    ----------
    G : NetworkX graph
    pred: dict
        Keyed by node to predecessor in the path
    dist: dict
        Keyed by node to the distance from the source
    source: node label
        Source node
    target: node label
        Target node
    weight: string
       Edge data key corresponding to the edge weight
    Returns
    -------
    length : numeric
        Length of a negative cycle if one exists.
        Otherwise, length of a shortest path.
        Length is inf if source and target are not connected.
    nodes: list
        List of nodes in a negative edge cycle (in order) if one exists.
        Otherwise, list of nodes in a shortest path.
        List is empty if source and target are not connected.
    negative_cycle : bool
        True if a negative edge cycle exists, otherwise False.
    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5, create_using = nx.DiGraph())
    >>> bf.bellman_ford(G, source=0, target=4)
    (3, [1, 2, 3, 4], False)
    """
    # Get shortest path tree
    pred, dist, negative_cycle_end = bellman_ford_tree(G, source, weight)

    nodes = []

    if negative_cycle_end:
        negative_cycle = True
        end = negative_cycle_end
        while True:
            nodes.insert(0, end)
            if nodes.count(end) > 1:
                end_index = nodes[1:].index(end) + 2
                nodes = nodes[:end_index]
                break
            end = pred[end]
    else:
        negative_cycle = False
        end = target
        while True:
            nodes.insert(0, end)
            # If end has no predecessor
            if pred.get(end, None) is None:
                # If end is not s, then there is no s-t path
                if end != source:
                    nodes = []
                break
            end = pred[end]

    if nodes:
        length = sum(
            G[u][v].get(weight, 1) for (u, v) in zip(nodes, nodes[1:])
        )
    else:
        length = float('inf')

    return length, nodes, negative_cycle


def bellman_ford_tree(G, source, weight='weight'):
    """
    Compute shortest path lengths and predecessors on shortest paths
    in weighted graphs using the Bellman-Ford algorithm.
    The algorithm has a running time of O(mn) where n is the number of
    nodes and m is the number of edges.  It is slower than Dijkstra but
    can handle negative edge weights.
    Parameters
    ----------
    G : NetworkX graph
       The algorithm works for all types of graphs, including directed
       graphs and multigraphs.
    source: node label
       Starting node for path
    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight
    Returns
    -------
    pred, dist : dictionaries
       Returns two dictionaries keyed by node to predecessor in the
       path and to the distance from the source respectively.
       Distance labels are invalid if a negative cycle exists.
    negative_cycle_end : node label
        Backtrack from this node using pred to find a negative cycle, if
        one exists; otherwise None.
    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5, create_using = nx.DiGraph())
    >>> pred, dist, negative_cycle_end = bf.bellman_ford_tree(G, 0)
    >>> sorted(pred.items())
    [(0, None), (1, 0), (2, 1), (3, 2), (4, 3)]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    >>> G = nx.cycle_graph(5, create_using = nx.DiGraph())
    >>> G[1][2]['weight'] = -7
    >>> bf_bellman_ford_tree(G, 0)
    ({0: 4, 1: 0, 2: 1, 3: 2, 4: 3}, {0: -12, 1: -11, 2: -15, 3: -14, 4: -13}, 0)
    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.
    The dictionaries returned only have keys for nodes reachable from
    the source.
    In the case where the (di)graph is not connected, if a component
    not containing the source contains a negative cost (di)cycle, it
    will not be detected.
    """
    if source not in G:
        raise KeyError("Node %s is not found in the graph" % source)

    dist = {source: 0}
    pred = {source: None}

    return _bellman_ford_relaxation(G, pred, dist, [source], weight)


def _bellman_ford_relaxation(G, pred, dist, source, weight):
    """
    Relaxation loop for Bellman–Ford algorithm
    Parameters
    ----------
    G : NetworkX graph
    pred: dict
        Keyed by node to predecessor in the path
    dist: dict
        Keyed by node to the distance from the source
    source: list
        List of source nodes
    weight: string
       Edge data key corresponding to the edge weight
    Returns
    -------
    pred, dist : dict
        Returns two dictionaries keyed by node to predecessor in the
        path and to the distance from the source respectively.
    negative_cycle_end : node label
        Backtrack from this node using pred to find a negative cycle, if
        one exists; otherwise None.
    """
    if G.is_multigraph():
        def get_weight(edge_dict):
            return min(eattr.get(weight, 1) for eattr in edge_dict.values())
    else:
        def get_weight(edge_dict):
            return edge_dict.get(weight, 1)

    G_succ = G.succ if G.is_directed() else G.adj
    inf = float('inf')
    n = len(G)

    count = {}
    q = deque(source)
    in_q = set(source)
    while q:
        u = q.popleft()
        in_q.remove(u)

        # Skip relaxations if the predecessor of u is in the queue.
        if pred[u] not in in_q:
            dist_u = dist[u]

            for v, e in G_succ[u].items():
                dist_v = dist_u + get_weight(e)

                if dist_v < dist.get(v, inf):
                    dist[v] = dist_v
                    pred[v] = u

                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0) + 1

                        if count_v == n:
                            negative_cycle_end = u
                            return pred, dist, negative_cycle_end

                        count[v] = count_v

    negative_cycle_end = None
    return pred, dist, negative_cycle_end



class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main():

    maze = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

    start = (7, 6)
    end = (1, 10)

    path = astar(maze, start, end)
    print(path)

main()


maze = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]]

start = (1, 1)
end = (6, 9)


path = astar(maze, start, end)

non_blocked_cells = []

for i in range(0,10):
  for j in range(0,10):
    if maze[i][j] == 0:
      non_blocked_cells.append([i+1,j+1]) 

import timeit      

total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for i in range(0,10,1):
    t = timeit.default_timer()
    start = non_blocked_cells[random.randint(0,70)]
    end = non_blocked_cells[random.randint(0,70)]
    if start == end:
      start = non_blocked_cells[random.randint(0,70)]
    else:
      path = astar(maze, start, end)
    elapsed_time = timeit.default_timer() - t
    total_time.append(elapsed_time)
vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
