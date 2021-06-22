import copy
from random import shuffle

import networkx as nx
import numpy as np
from scipy.special import softmax

from models.hs import RSL, HS


def euc_dis(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5


class UGA:
    def __init__(self, meta_graph=None):
        self.meta_graph = meta_graph

    def build_lists(self):
        self.overflow = [node for node in self.meta_graph.nodes() if self.meta_graph.nodes[node]['type'] == 'overflow']
        self.worker = [node for node in self.meta_graph.nodes() if self.meta_graph.nodes[node]['type'] == 'worker']
        self.underflow = [node for node in self.meta_graph.nodes() if
                          self.meta_graph.nodes[node]['type'] == 'underflow']

    def add_triplet(self, vertex, graph):
        neighbors = list(self.meta_graph.neighbors(vertex))
        shuffle(neighbors)
        overflow = None
        underflow = None
        for overflow_ in neighbors:
            if (not (overflow_ in list(graph.nodes()))) and self.meta_graph.nodes[overflow_]['type'] == 'overflow':
                overflow = overflow_
        for underflow_ in neighbors:
            if (not (underflow_ in list(graph.nodes()))) and self.meta_graph.nodes[underflow_]['type'] == 'underflow':
                underflow = underflow_
        if overflow is not None and underflow is not None:
            graph.add_node(overflow)
            graph.add_node(underflow)
            graph.add_node(vertex)
            graph.add_edge(vertex, underflow)
            graph.add_edge(vertex, overflow)

    # each species is a complete of worker station triples
    def build_matching(self):
        graph = nx.Graph()

        shuffle(self.worker)
        for w in self.worker:
            self.add_triplet(w, graph)
        graph.add_nodes_from([i for i in self.meta_graph.nodes if not i in list(graph.nodes)])
        return graph

    def find_overflow(self, vertex):
        while True:
            node_ind = np.random.choice(range(len(self.overflow)))
            if self.overflow[node_ind] != vertex:
                return self.overflow[node_ind]

    def find_underflow(self, vertex):
        while True:
            node_ind = np.random.choice(range(len(self.underflow)))
            if self.underflow[node_ind] != vertex:
                return self.underflow[node_ind]

    def mutate(self, worker, to_replace, graph):
        if self.meta_graph.nodes[to_replace]['type'] == 'overflow':
            other = self.find_overflow(to_replace)
        else:
            other = self.find_underflow(to_replace)
        if len(list(graph.neighbors(other))) > 0:
            otherw = list(graph.neighbors(other))[0]

            graph.add_edge(otherw, to_replace)
            graph.remove_edge(otherw, other)
        graph.add_edge(worker, other)
        graph.remove_edge(worker, to_replace)

    def pick_edge(self, ref_station, edges):
        for edge in edges:
            for vertex in edges:
                if self.meta_graph.nodes[vertex]['type'] == self.meta_graph.nodes[ref_station]['type']:
                    return edge

    def euc_fitness_ind(self, gene, spec):

        w = 0
        temp = {}
        neighbors = list(spec.neighbors(gene))
        if len(neighbors) != 2:
            return 0
        temp[self.meta_graph.nodes[neighbors[0]]['type']] = self.meta_graph.nodes[neighbors[0]]
        temp[self.meta_graph.nodes[neighbors[1]]['type']] = self.meta_graph.nodes[neighbors[1]]
        underflow_data = temp['underflow']
        overflow_data = temp['overflow']
        worker_data = self.meta_graph.nodes[gene]
        return euc_dis(worker_data['xe'], worker_data['ye'], underflow_data['x'],
                       underflow_data['y']) + euc_dis(worker_data['xs'], worker_data['ys'],
                                                      overflow_data['x'], overflow_data['y']) + euc_dis(
            underflow_data['x'], underflow_data['y'], overflow_data['x'], overflow_data['y'])

    def euc_fitness(self, spec):
        return -sum([self.euc_fitness_ind(gene, spec) for gene in spec.nodes if
                     self.meta_graph.nodes[gene]['type'] == 'worker'])

    def swap(self, worker, station, mate_station, spec):
        # print('worker',worker,'station',station,'mate_station',mate_station)
        # print('removed',worker,station)
        spec.remove_edge(worker, station)
        if len(list(spec.neighbors(mate_station))) > 0:
            unemployed = list(spec.neighbors(mate_station))[0]
            spec.remove_edge(unemployed, mate_station)
            spec.add_edge(worker, mate_station)
            pots = self.overflow if self.meta_graph.nodes[station]['type'] == 'overflow' else self.underflow
            shuffle(pots)
            for p in pots:
                if len(list(spec.neighbors(p))) == 0:
                    spec.add_edge(unemployed, p)
                    return
            print('NoStationFound')
        else:
            spec.add_edge(worker, mate_station)
            # print('added',worker,mate_station)

    def run(self, sswap_rate, oswap_rate, gens, pop_size, spec_opt=None):
        # make a population
        self.build_lists()
        pop = [self.build_matching() for i in range(pop_size)]
        scores = []
        bests = []
        for gen in range(gens):
            # get scores of species
            scores = [self.euc_fitness(s) for s in pop]
            bests.append(max(scores))
            # if gen %50 ==0 and update_flag:
            # print(min(scores),max(scores))
            print(scores)
            scores = softmax(scores)

            selected = []
            i = 0
            distr = []
            # while selecting pick a species
            while len(selected) < len(pop):
                spec_ind = np.random.choice(list(range(len(pop))), p=scores)
                # spec = pop[i]

                distr.append(spec_ind)
                spec_o = pop[spec_ind]
                spec = copy.deepcopy(spec_o)
                # for each gene (a triple)
                for j, node in enumerate(spec.nodes):
                    # swap with another pop
                    edges = set(spec.edges)
                    if self.meta_graph.nodes[node]['type'] != 'worker' and len(
                            list(spec.neighbors(node))) > 0 and np.random.random() < sswap_rate:
                        self.mutate(list(spec.neighbors(node))[0], node, spec)
                        # print(edges.intersection(set(self.meta_graph.edges).intersection(set(spec.edges))))
                for edge in spec.edges:
                    edge = list(edge)
                    if edge[0] in self.worker:
                        worker_, station_ = edge[0], edge[1]
                    else:
                        worker_, station_ = edge[1], edge[0]

                    if np.random.random() < oswap_rate:
                        mate_stations_ = list(pop[np.random.choice(list(range(len(pop))), p=scores)].neighbors(worker_))
                        mate_station_ = mate_stations_[0] if self.meta_graph.nodes[mate_stations_[0]]['type'] == \
                                                             self.meta_graph.nodes[station_]['type'] else \
                            mate_stations_[1]
                        self.swap(worker_, station_, mate_station_, spec)

                selected.append(spec)

            pop = [spec_opt(s) for s in selected]
        return pop[np.argmax(scores)], np.max(bests), bests

    def test(self, cwgraph):
        self.meta_graph = cwgraph
        return self.run(0.1, 0.1, 100, 100)


class UGA_RSL(UGA):
    def __init__(self, meta_graph=None):
        super().__init__(meta_graph)
        self.rsl = RSL()
        self.hs = HS()

    def build_matching(self):
        graph, score = self.rsl.optimize(self.meta_graph)

        for v1 in graph.nodes():
            for v2 in graph.nodes:
                if graph.has_edge(v1, v2) and {self.meta_graph.nodes[v1]['type'],
                                               self.meta_graph.nodes[v2]['type']} == {'overflow', 'underflow'}:
                    graph.remove_edge(v1, v2)
        graph.add_nodes_from([node for node in self.meta_graph.nodes if not graph.has_node(node)])
        print('RSL:', score, 'UGA:', self.euc_fitness(graph))
        return graph

    def opt_species(self, graph):
        graph2 = copy.deepcopy(graph)
        station_pairings = [(n1, n2) for n1 in graph.nodes for n2 in graph.nodes if
                            self.meta_graph.nodes[n1]['type'] == 'overflow' and self.meta_graph.nodes[n2][
                                'type'] == 'underflow']
        graph2.add_edges_from(station_pairings)
        graph3 = self.rsl.optimize(self.meta_graph, graph2)[0]
        graph3.remove_edges_from(station_pairings)
        return graph3

    def run_(self, oswap_rate, gens, pop_size):
        return super().run(0, oswap_rate, gens, pop_size, spec_opt=self.opt_species)

    def test(self, cwgraph):
        self.meta_graph = cwgraph
        return self.run_(0.1, 5, 5)
