from collections import defaultdict
from random import shuffle
import networkx as nx
from tqdm import tqdm

from graphprocessing import euc_dis
from models.trm import TRM


class Greed:
    def __init__(self, cwgraph=None, nodes=None, edges=None):
        self.cwgraph = cwgraph
        self.nodes = [] if nodes is None else nodes
        self.edges = [] if edges is None else edges

    def set_nodes(self, nodes):
        self.nodes = nodes

    def set_edges(self, edges):
        self.edges = edges

    def step(self):
        self.overflow = [node for node in self.cwgraph.nodes() if self.cwgraph.nodes[node]['type'] == 'overflow']
        self.worker = [node for node in self.cwgraph.nodes() if self.cwgraph.nodes[node]['type'] == 'worker']
        self.underflow = [node for node in self.cwgraph.nodes() if
                          self.cwgraph.nodes[node]['type'] == 'underflow']
        good_changes = []
        worker_delta = defaultdict(lambda: float('-inf'))
        print(len(self.edges))
        for e1, edge1 in enumerate(self.edges):
            ew = self.edge_weight(edge1)
            for e2, edge2 in enumerate(self.edges[e1:]):
                if edge1 != edge2:
                    ew2 = ew + self.edge_weight(edge2)
                    for i in range(3):
                        edge1t = [edge1[j] if i != j else edge2[i] for j in range(3)]
                        edge2t = [edge2[j] if i != j else edge1[i] for j in range(3)]
                        score = ew2 - self.edge_weight(edge1t) - self.edge_weight(edge2t)

                        if score > 1 and worker_delta[edge1t[0]] < score and worker_delta[edge2t[0]] < score:
                            topop = []
                            for j in range(len(good_changes)):
                                if e1 in good_changes[j] or e2 in good_changes[j]:
                                    topop.append(j)
                            for index in sorted(topop, reverse=True):
                                good_changes.pop(index)
                            good_changes.append([e1, e2, edge1t, edge2t])
                            worker_delta[edge1t[0]] = score
                            worker_delta[edge2t[0]] = score
        if len(good_changes) == 0:
            return False
        new_edges = [i[3] for i in good_changes] + [i[2] for i in good_changes]
        new_edges += [edge for e, edge in enumerate(self.edges) if not any([edge[0] in g for g in new_edges])]
        self.set_edges(new_edges)
        return True

    def edge_weight(self, edge):
        if edge[0] is None or edge[1] is None or edge[2] is None:
            return 0
        return euc_dis(self.cwgraph.nodes[edge[2]]['x'], self.cwgraph.nodes[edge[2]]['y'],
                       self.cwgraph.nodes[edge[1]]['x'],
                       self.cwgraph.nodes[edge[1]]['y']) + euc_dis(self.cwgraph.nodes[edge[0]]['xs'],
                                                                   self.cwgraph.nodes[edge[0]]['ys'],
                                                                   self.cwgraph.nodes[edge[1]]['x'],
                                                                   self.cwgraph.nodes[edge[1]]['y']) + euc_dis(
            self.cwgraph.nodes[edge[0]]['xe'], self.cwgraph.nodes[edge[0]]['ye'],
            self.cwgraph.nodes[edge[2]]['x'],
            self.cwgraph.nodes[edge[2]]['y'])

    def test(self, cwgraph):
        self.cwgraph = cwgraph
        self.make_trm_graph(cwgraph)
        i = 0
        while self.step():
            print(sum([self.edge_weight(edge) for edge in self.edges]))
            i += 1
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from([(edge[i], edge[j]) for i in range(3) for j in range(3) for edge in self.edges if i != j])
        return g, sum([self.edge_weight(edge) for edge in self.edges])

    def make_random_graph(self, cwgraph):
        overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
        underflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
        worker = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
        l = max([len(overflow), len(worker), len(underflow)])
        worker += [None] * (l - (len(worker)))
        underflow += [None] * (l - (len(underflow)))
        overflow += [None] * (l - (len(overflow)))
        shuffle(overflow)
        shuffle(underflow)
        shuffle(worker)
        self.set_nodes(worker + overflow + underflow)
        self.set_edges(list(zip(worker, overflow, underflow)))

    def make_trm_graph(self, cwgraph):
        trm = TRM()
        graph, score = trm.test(cwgraph)
        overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
        underflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
        worker = [str(node) for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
        print(len(graph.edges))
        edges = []
        for w in worker:
            for o in overflow:
                for u in underflow:
                    if graph.has_edge(w, o) and graph.has_edge(w, u)and graph.has_edge(o, u):
                        edges.append((w,o,u))
        print(len(edges))
        self.set_edges(edges)
        self.set_nodes(list(graph.nodes))
