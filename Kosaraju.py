from typing import List,Tuple


class Node:
    def __init__(self, value):
        self.value = value
        self.visited = False
        self.edges = []
        self.order = -1


class Graph:
    def __init__(self, edges: List[Tuple[int, int]], reversed=False):
        self.vertices = []
        self.graph = {}
        for edge in edges:
            if not reversed:
                self.add_edge(edge[0], edge[1])
            else:
                self.add_edge(edge[1], edge[0])
        self.curr_order = len(self.vertices)
        self.num_scc = 0
        self.scc = {}

    def add_vertex(self, value):
        if value not in self.graph:
            node = Node(value)
            self.graph[value] = node
            self.vertices.append(node)

    def get_vertex(self, value) -> Node:
        if value not in self.graph:
            self.add_vertex(value)
        return self.graph[value]

    def add_edge(self, from_value, to_value):
        from_node = self.get_vertex(from_value)
        to_node = self.get_vertex(to_value)
        from_node.edges.append(to_node)

    def topological_sort(self):
        for node in self.vertices:
            node.visited = False

        for node in self.vertices:
            if not node.visited:
                self.dfs(node.value)

    def dfs(self, value):
        start = self.graph[value]
        stack = [start]
        while len(stack):
            node = stack.pop()
            if not node.visited:
                node.visited = True
                for edge in node.edges:
                    stack.append(edge)
            elif node.order == -1:
                node.order = self.curr_order
                self.curr_order -= 1
        start.order = self.curr_order
        self.curr_order -= 1

    def dfs_scc(self, value):
        start = self.graph[value]
        if self.num_scc not in self.scc:
            self.scc[self.num_scc] = 1
        else:
            self.scc[self.num_scc] += 1
        stack = [start]
        while len(stack):
            node = stack.pop()
            if not node.visited:
                node.visited = True
                node.scc = self.num_scc
                for edge in node.edges:
                    stack.append(edge)


def kosaraju(edges: List[Tuple[int, int]]):
    graph = Graph(edges)

    graph_rev = Graph(edges, reversed=True)
    graph_rev.topological_sort()

    graph.num_scc = 0
    graph.vertices.sort(key=lambda v: graph_rev.graph[v.value].order, reverse=False)
    for node in graph.vertices:
        if not node.visited:
            graph.num_scc += 1
            graph.dfs_scc(node.value)

    print(list(sorted(graph.scc.values(), reverse=True))[:5])

if __name__ == '__main__':
    edges = [(1, 2), (3, 2), (4, 3), (1, 3), (4, 2)]
    # with open('problem8.10test1.txt', 'r') as file:
    #     for line in file.readlines():
    #         edge = line.strip().split(' ')
    #         if edge[0] != '':
    #             edges.append((int(edge[0]), int(edge[1])))

    kosaraju(edges)
