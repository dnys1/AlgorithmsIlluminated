from collections import deque
from typing import Dict, List, Tuple


def generate_edges(v: Dict[int, List[int]]):
    edges = []
    for vtx in v:
        for neighbor in v[vtx]:
            if (vtx, neighbor) not in edges and (neighbor, vtx) not in edges:
                edges.append((vtx, neighbor))
    return edges


class Vertex:
    def __init__(self, idx: int):
        self.idx = idx
        self.explored = False


class Graph:
    def __init__(self, v: Dict[int, List[int]]):
        self.v = {idx: Vertex(vtx) for vtx in v]}
        self.e = generate_edges(v)


def BFS(graph: Graph, s: int):
    graph.v[s] = True
    q = [s]
    while len(q) > 0:
        v = q.pop(0)
        if not v.explored:
            for edge in filter(lambda edge: edge[0] == v.idx or edge[1] == v.idx, graph.edges):
                edge.explored = True
                q.append(edge)


if __name__ == '__main__':
    graph = Graph({1: [2, 3, 4], 2: [3, 4, 5], 3: [4, 5, 1], 4: [3, 4, 2]})
    print(graph.v)
    print(graph.e)
