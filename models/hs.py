from random import shuffle

import networkx as nx
from networkx.algorithms import bipartite

from graphprocessing import edge_cost
from models.trm import TRM


class HS:

    def get_type(self, cwgraph):
        overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
        underflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
        workers = [str(node) for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
        return workers, overflow, underflow

    def constrain_graph(self, graph, constrain_type, cwgraph):
        # print(constrain_type)
        cost = 0
        fixed = nx.Graph()
        # print('num in:',len(graph.edges))
        worker, overflow, underflow = self.get_type(cwgraph)
        # print('worker_overflow'==constrain_type)
        if constrain_type == 'worker_overflow':
            # print( ' doing worker_overflow')
            for overflow_ in overflow:
                # print('outer')
                for worker_ in worker:
                    # print('inner')
                    # print(overflow_,str(worker_))

                    if graph.has_edge(overflow_, str(worker_)):
                        fixed.add_node(str(overflow_) + '|' + str(worker_), bipartite=0)

            for underflow_ in underflow:
                fixed.add_node(underflow_, bipartite=1)
        elif constrain_type == 'overflow_underflow':
            for underflow_ in underflow:
                for overflow_ in overflow:
                    if graph.has_edge(underflow_, overflow_):
                        fixed.add_node(underflow_ + '|' + overflow_, bipartite=0)
            for worker_ in worker:
                fixed.add_node(str(worker_), bipartite=1)
        elif constrain_type == 'underflow_worker':
            for underflow_ in underflow:
                for worker_ in worker:
                    if graph.has_edge(underflow_, str(worker_)):
                        fixed.add_node(str(underflow_) + '|' + str(worker_), bipartite=0)
            for overflow_ in overflow:
                fixed.add_node(overflow_, bipartite=1)

        solo_nodes = [node for node in fixed.nodes if fixed.nodes[node]['bipartite'] == 1]
        str_nodes = [node for node in fixed.nodes if fixed.nodes[node]['bipartite'] == 0]
        # print('str',len(str_nodes))
        # print('solo',len(solo_nodes))

        for node1 in str_nodes:
            for node2 in solo_nodes:
                if not fixed.has_edge(node1, node2):
                    fixed.add_edge(node1, node2, weight=edge_cost(*node1.split('|'), node2, cwgraph))
        # print('edges out', len(fixed.edges))
        # draw_bipartite(fixed)

        return bipartite.matching.minimum_weight_full_matching(fixed)

    def update_graph(self, matching):
        g = nx.Graph()
        for v1 in matching.keys():
            v2 = matching[v1]
            if not g.has_edge(v1, v2):
                vertices = v1.split('|') if '|' in v1 else [v1]
                vertices += v2.split('|') if '|' in v2 else [v2]
                g.add_nodes_from(vertices)
                g.add_edges_from([(v1_, v2_) for v1_ in vertices for v2_ in vertices if v1_ != v2_])
        return g

    def euc_dis(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5

    def matching_score(self, matching, graph):
        finished = set()
        w = 0
        for v in matching.nodes:
            if not (v in finished):
                neighbors = list(matching.neighbors(v))
                w += edge_cost(v, *neighbors, graph)
                finished = finished.union({v}).union(set(neighbors))
        return w

    def search(self, graph, cwgraph, max_rounds=float('inf'), threshold=0):
        best_score = float('inf')
        best_matching = None
        stalled_rounds = 0
        constrain_types = ['worker_overflow', 'overflow_underflow', 'underflow_worker']
        i = 0
        while stalled_rounds <= 1 and i < max_rounds:
            shuffle(constrain_types)
            for constrain_type in constrain_types:
                matching = self.constrain_graph(graph, constrain_type, cwgraph)
                graph = self.update_graph(matching)
            score = self.matching_score(graph, cwgraph)
            i += 1
            if score < best_score:
                if best_score - score > threshold:
                    stalled_rounds = 0
                best_score = score
                best_matching = graph

            else:
                stalled_rounds += 1
        return best_matching, best_score


class RLS:
    def make_graph(self, cwgraph):
        overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
        underflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
        workers = [str(node) for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
        shuffle(overflow)
        shuffle(underflow)
        shuffle(workers)
        full = min([len(overflow), len(underflow), len(workers)])
        overflow = overflow[:full]
        underflow = underflow[:full]
        workers = workers[:full]

        wou_triplets = [(w, o, u) for w, o, u in zip(workers, overflow, underflow)]
        graph = nx.Graph()

        for w, o, u in wou_triplets:
            graph.add_nodes_from([w, o, u])
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
        return self.optimize(cwgraph)


class TRM_RLS(RLS):
    def make_graph(self, cwgraph):
        trm = TRM()
        return trm.test(cwgraph)[0]


class RLSL(RLS):
    def test(self, cwgraph):
        graph = self.make_graph(cwgraph)
        hs = HS()
        return hs.search(graph, cwgraph, max_rounds=2)
