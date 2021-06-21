import numpy as np
import networkx as nx
from scipy.special import softmax
from tqdm import tqdm
from random import shuffle
import copy
from models.hs import RSL, HS


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
            if (not overflow_ in list(graph.nodes())) and self.meta_graph.nodes[overflow_]['type'] == 'overflow':
                overflow = overflow_
        for underflow_ in neighbors:
            if (not underflow_ in list(graph.nodes())) and self.meta_graph.nodes[underflow_]['type'] == 'underflow':
                underflow = underflow_
        if overflow != None and underflow != None:
            graph.add_node(overflow)
            graph.add_node(underflow)
            graph.add_node(vertex)
            graph.add_edge(vertex, underflow)
            graph.add_edge(vertex, overflow)

    def euc_dis(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5

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
        neighbors = list(spec.neighbors(gene))
        if len(neighbors) != 0:
            for i in neighbors:
                if self.meta_graph.nodes[i]['type'] == 'underflow':
                    w += self.euc_dis(self.meta_graph.nodes[i]['x'], self.meta_graph.nodes[i]['y'],
                                      self.meta_graph.nodes[gene]['xs'], self.meta_graph.nodes[gene]['ys'])
                else:
                    w += self.euc_dis(self.meta_graph.nodes[i]['x'], self.meta_graph.nodes[i]['y'],
                                      self.meta_graph.nodes[gene]['xe'], self.meta_graph.nodes[gene]['ye'])
            w += self.euc_dis(self.meta_graph.nodes[neighbors[0]]['x'], self.meta_graph.nodes[neighbors[0]]['y'],
                              self.meta_graph.nodes[neighbors[1]]['x'], self.meta_graph.nodes[neighbors[1]]['y'])

        return w

    def euc_fitness(self, spec):
        return -sum([self.euc_fitness_ind(gene, spec) for gene in spec.nodes if
                     self.meta_graph.nodes[gene]['type'] == 'worker'])

    def swap(self, worker, station, mate_station, spec):
        # print('worker',worker,'station',station,'mate_station',mate_station)
        # print('removed',worker,station)
        spec.remove_edge(worker, station)
        if len(list(spec.neighbors(mate_station))) > 0:
            unemployed = list(spec.neighbors(mate_station))[0]
            # print('mate_neighbors:', list(spec.neighbors(mate_station)))
            # print('worker:', worker)
            # print('unemplowed:',unemployed)
            spec.remove_edge(unemployed, mate_station)
            # print('removed',unemployed,mate_station)
            # print('added',worker,mate_station)
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

        bests = []
        for gen in range(gens):
            # get scores of species
            scores = np.array([self.euc_fitness(s) for s in pop])
            bests.append(max(scores))
            # if gen %50 ==0 and update_flag:
            # print(min(scores),max(scores))
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
            # print(distr)
        return pop[np.argmax(scores)], np.max(bests), bests

    def test(self, cwgraph):
        self.meta_graph = cwgraph
        return self.run(0.1, 0.1, 100, 100)


class UGA_RSL(UGA):
    def __init__(self, meta_graph=None):
        super().__init__(meta_graph)
        self.rsl = RSL()
        self. hs = HS()
    def build_matching(self):
        # print('g')
        graph = self.rsl.optimize(self.meta_graph)[0]
        graph.add_nodes_from([node for node in self.meta_graph.nodes if not graph.has_node(node)])
        for v1 in graph.nodes():
            for v2 in graph.nodes:
                if graph.has_edge(v1, v2) and set(
                        [self.meta_graph.nodes[v1]['type'], self.meta_graph.nodes[v2]['type']]) == set(
                        ['overflow', 'underflow']):
                    graph.remove_edge(v1, v2)
        return graph

    def opt_species(self,graph):
        return self.hs.seach(graph,self.meta_graph)

    def run(self, oswap_rate, gens, pop_size):
        return super().run(0, oswap_rate, gens, pop_size,spec_opt=self.opt_species)

    def test(self, cwgraph):
        self.meta_graph = cwgraph
        return self.run(0.1, 5, 5)
