import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from models.trm import TRM
from graphprocessing import types, partial_scorer
from itertools import combinations
class Graph3Opt:
    """
    Takes a graph and returns the graph with types changed according to the change_dict 
    """
    def change_graph_types(self, graph, change_dict):
        for node in graph.nodes():
            graph.nodes[node]['type'] = change_dict[graph.nodes[node]['type']]
        return graph

    
    def solve_trm(self, graph):
        """
        Takes a graph and optimizes it according to TRM
        """
        trm = TRM(graph)
        trm.solve()
        return trm.get_result()[0]
    
    def __init__(self):
        self.best_score = float('inf')
        self.best_graph = None
    
    def assemble_optimal_graph(self, graph, edges):
        """
        Recursively builds optimal graph solutions by building all possible solution graphs adding edges one at a time 
        Removes solutions that are greater than the best score
        Best score is updated whenever a graph is completed
        """
        if len(edges) == 0:
            if self.best_score > partial_scorer(graph):
                self.best_graph = graph
                self.best_score = partial_scorer(graph)
        
        for edge in edges:
            new_graph = graph.copy()
            new_graph.add_edge(edge[0], edge[1])
            if partial_scorer(graph) > self.best_score:
                continue
            self.assemble_optimal_graph(new_graph, edges.remove(edge))
        
    def solve(self,cwgraph):
        """
        Builds 3 different TRM optimized graphs and uses TRM graphs to create set of all their edges
        Then builds graph from the combined edge set
        """
        change_dicts = [{types[i]:c[i] for i in range(3)} for c in combinations(types)]
        graphs = [self.change_graph_types(cwgraph, change_dict) for change_dict in change_dicts]
        edges = set()
        for graph in graphs:
            edges |= set(graph.edges())
        self.assemble_optimal_graph(cwgraph, edges)
        return self.best_graph, self.best_score
    
    def test(self,cwgraph):
        return self.solve(cwgraph)