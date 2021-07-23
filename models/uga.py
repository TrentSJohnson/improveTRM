import copy
from multiprocessing import Pool
from random import shuffle
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from graphprocessing import complete_graph, score_graph, get_entities
from models.hs import HS


class UGA:
    def __init__(self, meta_graph=None):
        self.meta_graph = meta_graph
        self.overflows = []
        self.underflows = []
        self.workers = []
        self.num_triplets = 0

    def build_lists(self):
        self.workers, self.overflows, self.underflows = get_entities(self.meta_graph)

    def build_matching(self, _=None):
        graph = nx.Graph()
        ws = self.workers.copy()
        os = self.overflows.copy()
        us = self.underflows.copy()
        shuffle(os)
        shuffle(us)
        shuffle(ws)
        m = min(len(self.overflows), len(self.workers), len(self.underflows))
        graph.add_nodes_from(self.meta_graph)
        for w, o, u in zip(ws[:m], os[:m], us[:m]):
            graph.add_edge(w, o)
            graph.add_edge(w, u)
        return graph

    def euc_fitness(self, spec):
        spec_ = complete_graph(spec.copy(), self.meta_graph)
        return -score_graph(spec_, self.meta_graph)

    def graph_to_list(self, graph):
        genome = []
        for w in self.workers:
            neighbors = list(graph.neighbors(w))
            if len(neighbors) == 2:
                o = neighbors[0] if self.meta_graph.nodes[neighbors[0]]['type'] == 'overflow' else neighbors[1]
                u = neighbors[0] if self.meta_graph.nodes[neighbors[0]]['type'] == 'underflow' else neighbors[1]
            elif len(neighbors) == 1:
                o = neighbors[0] if self.meta_graph.nodes[neighbors[0]]['type'] == 'overflow' else 'odummy' + w
                u = neighbors[0] if self.meta_graph.nodes[neighbors[0]]['type'] == 'underflow' else 'udummy' + w
            else:
                o = 'odummy' + w
                u = 'udummy' + w
            genome += [o, u]
        return genome

    def list_to_graph(self, genome):
        graph = nx.Graph()
        graph.add_nodes_from(self.meta_graph.nodes)
        for w, worker in enumerate(self.workers):
            if not ('dummy' in genome[2 * w]):
                graph.add_edge(worker, genome[w * 2])
            if not ('dummy' in genome[2 * w + 1]):
                graph.add_edge(worker, genome[w * 2 + 1])
        return graph

    def pmx(self, parent1, parent2):
        data = list(set(parent1 + parent2))
        parent1 = np.array([data.index(g) for g in parent1])
        parent2 = np.array([data.index(g) for g in parent2])
        if len(parent1) != pd.Series(parent1).nunique() or len(parent2) != pd.Series(parent2).nunique():
            print(parent1, parent2)
        child1 = np.array([np.nan] * len(parent1))
        child2 = np.array([np.nan] * len(parent1))
        start, stop = sorted([np.random.randint(0, len(parent1)), np.random.randint(0, len(parent2))])
        while start == stop:
            start, stop = sorted([np.random.randint(0, len(parent1)), np.random.randint(0, len(parent2))])
        map1 = parent1[start:stop]
        map2 = parent2[start:stop]
        cut = np.array(range(start, stop))
        child1[cut] = map2
        child2[cut] = map1
        mapping = [{n1: n2, n2: n1} for n1, n2 in zip(map1, map2)]
        for parent, child in [(parent1, child1), (parent2, child2)]:
            for i, node in enumerate(parent):
                if not (i in cut):
                    if not (node in child):
                        child[i] = node
                    else:
                        guess = node
                        used_maps = []
                        while guess in child:
                            map = {}
                            finding = True
                            for map in mapping:
                                if finding and guess in map.keys() and not (map in used_maps):
                                    guess = map[guess]
                                    finding = False
                                    used_maps.append(map)
                            # print('in while')
                        child[i] = guess
                        # print('left while')
        if len(child1) != pd.Series(child1).nunique() or len(child2) != pd.Series(child2).nunique():
            print(child1, child2)
        child1 = [data[int(g)] for g in child1]
        child2 = [data[int(g)] for g in child2]
        return child1, child2

    def temp(self, n1, n2):
        return n1['name'] == n2['name']

    def add_names(self, pop):
        for spec in pop:
            nx.set_edge_attributes(spec, {edge: {'name': '_'.join(edge)} for edge in list(spec.edges)})
            nx.set_node_attributes(spec, self.meta_graph.nodes)

    def detect_duplicate(self, spec, pop):
        specl = self.graph_to_list(spec)
        if 0 == pop.index(spec):
            return True
        references = pop[:pop.index(spec)]
        for ref in references:
            if all([s == p for s, p in zip(specl, self.graph_to_list(ref))]):
                return True
        return False

    def run(self, gens, pop_size, spec_opt=None):
        # make a population
        self.build_lists()
        pop = [self.build_matching() for i in range(pop_size)]
        self.add_names(pop)
        scores = [self.euc_fitness(complete_graph(score, self.meta_graph)) for score in pop]

        scores = np.array(scores) / sum(scores)
        best_score = float('-inf')
        best_graph = pop[0]
        for gen in tqdm(range(gens)):
            # get scores of species
            print("Pop Size", len(pop))

            selected = []
            if len(pop) == 1:
                print("UGA RSL Failed Pop at gen", gen)
                return best_graph, best_score
            # while selecting pick a species
            for i in range(pop_size):
                ind1 = np.random.choice(list(range(len(pop))), p=scores)
                ind2 = np.random.choice(list(range(len(pop))), p=scores)
                gparent1 = pop[ind1].copy()
                gparent2 = pop[ind2].copy()
                parent1 = self.graph_to_list(gparent1)
                parent2 = self.graph_to_list(gparent2)
                child1, child2 = self.pmx(parent1, parent2)
                gchild1 = self.list_to_graph(child1)
                gchild2 = self.list_to_graph(child2)
                if len(self.graph_to_list(gchild1)) != pd.Series(self.graph_to_list(gchild1)).nunique():
                    print(gchild1)
                selected.append(gchild1)
                if len(self.graph_to_list(gchild2)) != pd.Series(self.graph_to_list(gchild2)).nunique():
                    print(gchild2)

                selected.append(gchild2)
            pool = []
            for spec in selected:
                if not self.detect_duplicate(spec, pop):
                    pool.append(spec)
            pool = sorted(pool, key=lambda x: self.euc_fitness(complete_graph(x)))
            pop = pool[:min([len(pool), pop_size])]
            # check for duplicates
            if not (spec_opt is None):
                print('optimizing')
                with Pool(processes=8) as pool:
                    pop = pool.map(spec_opt, pop)
                print('ending optimizinng')

            for spec in pop:
                if len(self.graph_to_list(spec)) != pd.Series(self.graph_to_list(spec)).nunique():
                    print(spec)
            self.add_names(pop)
            scores = [self.euc_fitness(complete_graph(graph, self.meta_graph)) for graph in pop]
            scores = np.array(scores) / sum(scores)
            if max(scores) > best_score:
                best_graph = complete_graph(pop[np.argmax(scores)], self.meta_graph)
                best_score = max(scores)
            if len(pop) < pop_size:
                print('population size less than population size')
                return best_graph, best_score
                    

        return best_graph, best_score

    def test(self, cwgraph):
        self.meta_graph = cwgraph
        self.workers, self.overflows, self.underflows = get_entities(cwgraph)
        return self.run(gens=4, pop_size=12)


class UGA_RLS(UGA):
    def __init__(self, meta_graph=None):
        super().__init__(meta_graph)
        self.hs = HS()

    def opt_species(self, graph):
        graph2 = copy.deepcopy(graph)
        station_pairings = [(n1, n2) for n1 in graph.nodes for n2 in graph.nodes if
                            self.meta_graph.nodes[n1]['type'] == 'overflow' and self.meta_graph.nodes[n2][
                                'type'] == 'underflow']
        graph2.add_edges_from(station_pairings)
        graph3 = self.hs.search(graph2, self.meta_graph)[0]
        graph3.remove_edges_from(station_pairings)
        graph3.add_nodes_from([node for node in self.meta_graph.nodes if not graph3.has_node(node)])
        return graph3

    def test(self, cwgraph, gens=4, pop_size=10):
        self.meta_graph = cwgraph
        return self.run(gens=gens, pop_size=pop_size, spec_opt=self.opt_species)
