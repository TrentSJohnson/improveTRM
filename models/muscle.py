import networkx as nx
from networkx.algorithms import bipartite

from graphprocessing import types, partial_scorer, edge_cost, edge2_cost
from models.trm import TRM


class Graph3Opt:

    def solve_trm(self, graph, cwgraph):
        """
        Takes a graph and optimizes it according to TRM
        """
        trm = TRM()
        trm.solve(graph, cwgraph)
        return trm.get_result()[0]

    def __init__(self):
        self.best_score = float('inf')
        self.best_graph = None

    def assemble_optimal_graph(self, graph, edges, cwgraph, n_workers, max_depth, depth=0, path=0, partial_score=0):
        """
        Recursively builds optimal graph solutions by building all possible solution graphs adding edges one at a time 
        Removes solutions that are greater than the best score
        Best score is updated whenever a graph is completed
        """
        if n_workers > len(graph.nodes) + len(edges):
            return
        if len(edges) == 0:
            ps = partial_scorer(graph, cwgraph)
            if self.best_score > ps:
                self.best_graph = graph
                self.best_score = ps
                print('new best: ', path/(2**max_depth)*100,'% done')

            return
        edge = edges[0]
        if len(set(edge).intersection(graph.nodes)) == 0:
            new_graph = graph.copy()
            new_graph.add_nodes_from(edge)
            new_graph.add_edge(edge[0], edge[1])
            ps = edge2_cost(*edge, cwgraph) + partial_score
            if ps < self.best_score:
                self.assemble_optimal_graph(new_graph, edges[1:], cwgraph, n_workers, max_depth, depth+1, path, ps)
        self.assemble_optimal_graph(graph, edges[1:], cwgraph, n_workers, max_depth, depth+1, path+(2**(max_depth-depth)))

    def build_constrained_graph(self, matching, cwgraph):
        """
        Args:
            matching: dict
                contains dict with vertices as keys and their matching vertex as values
            cwgraph: nx.Graph
                graph of vetex data

        Returns: nx.Graph
            graph with combined matching vertices, singel vertices and edges with weights

        """
        single_type = list(set(types) - set([cwgraph.nodes[node]['type'] for node in matching.keys()]))[0]
        new_graph = nx.Graph()
        for w in matching.keys():
            new_graph.add_node(w + '_' + matching[w])
        double = [k + '_' + v for k, v in zip(matching.keys(), matching.values())]
        single = [u for u in cwgraph.nodes if cwgraph.nodes[u]['type'] == single_type]
        new_graph.add_nodes_from(single)
        for s in single:
            for d in double:
                new_graph.add_edge(s, d, weight=edge_cost(s, *(d.split('_')), cwgraph))
        return new_graph

    def build_limited_graph(self, cwgraph, type1, type2):
        """
        cwgraph: graph of data
        type1: frist type of nodes from cwgraph to include in new graph
        type2: second type of nodes form cwgraph to include in new graph

        returns: a nx.Graph with vertices from cwgraph of type1 and type2 and edges costs from edge2_cost
        """
        new_graph = nx.Graph()
        for n1 in cwgraph.nodes():
            if cwgraph.nodes[n1]['type'] == type1:
                for n2 in cwgraph.neighbors(n1):
                    if cwgraph.nodes[n2]['type'] == type2:
                        new_graph.add_nodes_from([n1, n2])
                        new_graph.add_edge(n1, n2, weight=edge2_cost(n1, n2, cwgraph))
        return new_graph

    def decontrain_matching(self, con_matching, cwgraph):
        """
        Split apart contrianed vertices (vertices with '_')
        args:
            con_matching: dict
            cwgraph: nx.Graph
        returns:
            nx.Graph
        """
        new_graph = nx.Graph()
        finished = set()
        for u, v in con_matching.items():
            if not(u in finished or v in finished):
                vs = u.split('_') + v.split('_')
                new_graph.add_nodes_from(vs)
                new_graph.add_edge(vs[0], vs[1], weight=edge_cost(*vs, cwgraph))
                new_graph.add_edge(vs[2], vs[1], weight=edge_cost(*vs, cwgraph))
                new_graph.add_edge(vs[0], vs[2], weight=edge_cost(*vs, cwgraph))
                finished |= set(vs)
        return new_graph

    def solve(self, cwgraph):
        """
        Builds 3 different TRM optimized graphs and uses TRM graphs to create set of all their edges
        Then builds graph from the combined edge set
        """
        self.best_score = float('inf')
        self.best_graph = None
        graphs = [self.build_limited_graph(cwgraph, 'worker', 'overflow'),
                  self.build_limited_graph(cwgraph, 'worker', 'underflow'),
                  self.build_limited_graph(cwgraph, 'overflow', 'underflow')]
        matchings = [bipartite.matching.minimum_weight_full_matching(graph) for graph in graphs]

        con_graphs = [self.build_constrained_graph(m, cwgraph) for m in matchings]
        con_matchings = [bipartite.minimum_weight_full_matching(g) for g in con_graphs]
        matched_graphs = [self.decontrain_matching(m, cwgraph) for m in con_matchings]
        edges = set()
        for graph in matched_graphs:
            edges |= set(graph.edges())
        edges = list(edges)
        nodes = set()
        for g in graphs:
            nodes |= set(g.nodes)
        n_workers = len([1 for i in list(nodes) if cwgraph.nodes[i]['type'] == 'worker'])
        n_graphs = 2**(2*n_workers)
        self.assemble_optimal_graph(nx.Graph(), edges, cwgraph,n_workers, n_graphs)
        return self.best_graph, self.best_score

    def test(self, cwgraph):
        return self.solve(cwgraph)
