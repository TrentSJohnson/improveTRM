from random import shuffle

import networkx as nx
from networkx.algorithms import bipartite

from graphprocessing import complete_graph, edge_cost, get_entities, score_graph
from models.trm import TRM
import pandas as pd


class HS:
    def __init__(self):
        self.workers = None
        self.overflows = None
        self.underflows = None

    def worker_neighbor(self, node, graph, cwgraph):
        if node in self.workers:
            return node
        for n in graph.neighbors(node):
            if n in self.workers:
                return n
        return -1

    def constrain_graph(self, graph, constrain_type, cwgraph):
        """
        Takes in a graph and combines nodes edges based on the a constrain type and returns and optimal matching of the contrained graph.
        Args:
            graph: nx.Graph
                A matching graph
            constrain_type: str
                A string indicating the constrain type
            cwgraph: nx.Graph
                The cwgraph
        Returns:
            matching: dict
        """
        fixed = nx.Graph()
        workers, overflows, underflows = get_entities(cwgraph)
        if constrain_type == 'worker_overflow':
            nodes1 = workers
            nodes2 = overflows
            nodes3 = underflows
        elif constrain_type == 'overflow_underflow':
            nodes1 = overflows
            nodes2 = underflows
            nodes3 = workers
        else:
            nodes1 = underflows
            nodes2 = workers
            nodes3 = overflows

        for node1 in nodes1:
            for node2 in nodes2:
                w1 = self.worker_neighbor(node1, graph, cwgraph)
                w2 = self.worker_neighbor(node2, graph, cwgraph)
                if w1 == w2 and w1 != -1:
                    fixed.add_node(str(node1) + '|' + str(node2), bipartite=0)
        for node3 in nodes3:
            fixed.add_node(node3, bipartite=1)

        solo_nodes = [node for node in fixed.nodes if fixed.nodes[node]['bipartite'] == 1]
        comb_nodes = [node for node in fixed.nodes if fixed.nodes[node]['bipartite'] == 0]
        for comb in comb_nodes:
            for solo in solo_nodes:
                fixed.add_edge(comb, solo, weight=edge_cost(*comb.split('|'), solo, cwgraph))
        return bipartite.matching.minimum_weight_full_matching(fixed)

    def decontr_matching(self, contr_matching, cwgraph):
        g = nx.Graph()
        g.add_nodes_from(self.workers + self.overflows + self.underflows)
        finished = []
        for v1 in contr_matching.keys():
            v2 = contr_matching[v1]
            vertices = v1.split('|') if '|' in v1 else [v1]
            vertices += v2.split('|') if '|' in v2 else [v2]
            if not (vertices[0] in finished):
                finished += vertices
                g.add_edges_from([(v1_, v2_) for v1_ in vertices for v2_ in vertices if
                                  cwgraph.nodes[v1_]['type'] == 'worker' and cwgraph.nodes[v2_]['type'] != 'worker'])
        return g

    def search(self, graph, cwgraph, max_stalled_rounds=1, max_rounds=float('inf'), threshold=0):
        self.workers, self.overflows, self.underflows = get_entities(cwgraph)

        best_score = score_graph(complete_graph(graph, cwgraph), cwgraph)
        best_matching = graph
        stalled_rounds = 0
        constrain_types = ['worker_overflow', 'overflow_underflow', 'underflow_worker']
        i = 0
        updating = True
        already_updated = False
        while updating:
            shuffle(constrain_types)
            already_updated = False
            updating = False
            graph_, degraph = None, best_matching
            if len(self.graph_to_list(graph, cwgraph)) != pd.Series(self.graph_to_list(graph, cwgraph)).nunique():
                print(degraph)
            for constrain_type in constrain_types:
                old_degraph = degraph.copy()
                constr_matching = self.constrain_graph(degraph, constrain_type, cwgraph)
                degraph = self.decontr_matching(constr_matching, cwgraph)
                # degraph.add_nodes_from(get_entities(cwgraph)[0])
                list_ = self.graph_to_list(degraph, cwgraph)
                ulist_ = pd.Series(list_).unique()
                if len(list_) != len(ulist_):
                    print(degraph)
                # list_ = self.graph_to_list(degraph,cwgraph)
                # if len(list_) != pd.Series(list_).nunique():
                #    print(degraph)
            graph_ = complete_graph(degraph, cwgraph)
            score = score_graph(graph_, cwgraph)
            if score < best_score:
                best_score = score
                best_matching = degraph
                updating = True
            i += 1
        return best_matching, best_score

    def graph_to_list(self, graph, cwgraph):
        genome = []
        for w in self.workers:
            neighbors = list(graph.neighbors(w))
            if len(neighbors) == 2:
                o = neighbors[0] if cwgraph.nodes[neighbors[0]]['type'] == 'overflow' else neighbors[1]
                u = neighbors[0] if cwgraph.nodes[neighbors[0]]['type'] == 'underflow' else neighbors[1]
            elif len(neighbors) == 1:
                o = neighbors[0] if cwgraph.nodes[neighbors[0]]['type'] == 'overflow' else 'odummy' + w
                u = neighbors[0] if cwgraph.nodes[neighbors[0]]['type'] == 'underflow' else 'udummy' + w
            else:
                o = 'odummy' + w
                u = 'udummy' + w
            genome += [o, u]

        return genome


class RLS:
    def make_graph(self, cwgraph):
        workers, overflow, underflow = get_entities(cwgraph)
        m = min((len(workers), len(overflow), len(underflow)))
        wou_triplets = [(w, o, u) for w, o, u in zip(workers[:m], overflow[:m], underflow[:m])]
        graph = nx.Graph()
        graph.add_nodes_from(cwgraph.nodes)
        for w, o, u in wou_triplets:
            graph.add_edge(w, u)
            graph.add_edge(u, o)
            graph.add_edge(w, o)
        return graph

    def optimize(self, cwgraph, graph=None):
        if graph is None:
            graph = self.make_graph(cwgraph)
        hs = HS()
        return hs.search(graph, cwgraph)

    def test(self, cwgraph):
        graph = self.make_graph(cwgraph)
        hs = HS()
        return hs.search(graph, cwgraph)


class TRM_RLS(RLS):
    def make_graph(self, cwgraph):
        trm = TRM()
        return trm.test(cwgraph)[0]

    def test(self, cwgraph):
        graph = self.make_graph(cwgraph)
        hs = HS()
        return hs.search(graph, cwgraph)


class RLSL(RLS):
    def test(self, cwgraph):
        graph = self.make_graph(cwgraph)
        hs = HS()
        return hs.search(graph, cwgraph, max_rounds=2)
