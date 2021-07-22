import copy
from multiprocessing import Pool
from random import shuffle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from graphprocessing import complete_graph, score_graph, get_entities
from models.hs import RLS, HS


def euc_dis(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5


class UGA:
    def __init__(self, meta_graph=None):
        self.meta_graph = meta_graph
        self.overflows = []
        self.underflows = []
        self.workers = []
        self.num_triplets = 0

    def build_lists(self):
        self.workers, self.overflows, self.underflows = get_entities(self.meta_graph)


    # each species is a complete of workers station triples
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
            print(parent1,parent2)
        child1 = np.array([np.nan] * len(parent1))
        child2 = np.array([np.nan] * len(parent1))
        start, stop = sorted([np.random.randint(0, len(parent1)), np.random.randint(0, len(parent2))])
        while start == stop:
            start, stop = sorted([np.random.randint(0, len(parent1)), np.random.randint(0, len(parent2))])
        map1 = parent1[start:stop]
        map2 = parent2[start:stop]
        cut = np.array(range(start,stop))
        child1[cut]=map2
        child2[cut]=map1
        mapping = [{n1:n2,n2:n1} for n1,n2 in zip(map1,map2)]
        for parent, child in [(parent1,child1),(parent2,child2)]:
            for i, node in enumerate(parent):
                if not (i in cut):
                    if not (node in child):
                        child[i] = node
                    else:
                        guess = node
                        used_maps= []
                        while guess in child:
                            map = {}
                            finding = True
                            for map in mapping:
                                if finding and guess in map.keys() and not(map in used_maps):
                                    guess = map[guess]
                                    finding = False
                                    used_maps.append(map)
                            #print('in while')
                        child[i] = guess
                        #print('left while')
        if len(child1) != pd.Series(child1).nunique() or len(child2) != pd.Series(child2).nunique():
                print(child1, child2)
        child1 = [data[int(g)] for g in child1]
        child2 = [data[int(g)] for g in child2]
        return child1, child2


    def pmx___(self, parent1, parent2, r_a_b=None, ):
        data = list(set(parent1+parent2))
        parent1 = [data.index(g) for g in parent1]
        parent2 = [data.index(g) for g in parent2]
        if len(parent1) != pd.Series(parent1).nunique() or len(parent2) != pd.Series(parent2).nunique():
            print(parent1,parent2)
        parent1 = np.array(parent1)
        parent2 = np.array(parent2)
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        a, b = sorted([np.random.randint(0, len(parent1)), np.random.randint(0, len(parent2))])
        length = len(parent1)
        min_a_b, max_a_b = min([a, b]), max([a, b])
        if r_a_b is None:
            r_a_b = range(min_a_b, max_a_b)
        r_left = np.delete(range(length), r_a_b)
        left_1, left_2 = child1[r_left], child2[r_left]
        middle_1, middle_2 = child1[r_a_b], child2[r_a_b]
        child1[r_a_b], child2[r_a_b] = middle_2, middle_1
        mapping = [[], []]
        for i, j in zip(middle_1, middle_2):
            if j in middle_1 and i not in middle_2:
                index = np.argwhere(middle_1 == j)[0, 0]
                value = middle_2[index]
                while True:
                    if value in middle_1:
                        index = np.argwhere(middle_1 == value)[0, 0]
                        value = middle_2[index]
                    else:
                        break
                mapping[0].append(i)
                mapping[1].append(value)
            elif i in middle_2:
                pass
            else:
                mapping[0].append(i)
                mapping[1].append(j)
        for i, j in zip(mapping[0], mapping[1]):
            if i in left_1:
                left_1[np.argwhere(left_1 == i)[0, 0]] = j
            elif i in left_2:
                left_2[np.argwhere(left_2 == i)[0, 0]] = j
            if j in left_1:
                left_1[np.argwhere(left_1 == j)[0, 0]] = i
            elif j in left_2:
                left_2[np.argwhere(left_2 == j)[0, 0]] = i
        child1[r_left], child2[r_left] = left_1, left_2
        if len(child1) != pd.Series(child1).nunique() or len(child2) != pd.Series(child2).nunique():
            print(child1, child2)
        child1 = [data[g] for g in child1]
        child2 = [data[g] for g in child2]


        return child1, child2



    def pmx__(self, parent1, parent2):
        """
                Implementation of Partially Mapped Crossover (PMX)
                """
        child1 = [None] * len(parent1)
        child2 = [None] * len(parent1)
        start, stop = sorted([np.random.randint(0, len(parent1)), np.random.randint(0, len(parent2))])
        map1 = parent1[start:stop]
        map2 = parent2[start:stop]
        for i in range(start,stop):
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        if start != 0:
            for i in range(start):
                if not (parent1[i] in child1):
                    child1[i] = parent1[i]
                if not (parent2[i] in child2):
                    child2[i] = parent2[i]
        if stop != len(child1):
            for i in range(stop,len(child1)):
                if not (parent1[i] in child1):
                    child1[i] = parent1[i]
                if not (parent2[i] in child2):
                    child2[i] = parent2[i]

        pairs=[]
        for m1, m2 in zip(map1, map2):
            if m1 != m2:
                pairs.append({m1, m2})

        for i, node in enumerate(child1):
            if node is None:
                done_pairs = []
                guess = parent1[i]
                while guess in child1:
                    pair = [pair for pair in pairs if guess in pair and not(pair in done_pairs)][0]
                    done_pairs.append(pair)
                    guess = list(set(pair)-{guess})[0]
                child1[i] = guess
        for i, node in enumerate(child2):
            if node is None:
                done_pairs = []
                guess = parent2[i]
                while guess in child2:
                    pair = [pair for pair in pairs if guess in pair and not(pair in done_pairs)][0]
                    done_pairs.append(pair)
                    guess = list(set(pair)-{guess})[0]
                child2[i] = guess
        return child1, child2

    def pmx_(self, parent1, parent2):
        """
        Implementation of Partially Mapped Crossover (PMX)
        """
        child1 = [None]*len(parent1)
        child2 = [None]*len(parent1)
        start, stop = sorted([np.random.randint(0, len(parent1)), np.random.randint(0, len(parent2))])
        map1 = parent1[start:stop]
        map2 = parent2[start:stop]
        pairs = []
        for m1, m2 in zip(map1, map2):
            if m1 != m2:
                pairs.append({m1, m2})
        safe_mappings = {}
        unsafe_pairs = []
        unsafe_nodes = []
        unsafe_node = None
        for pair in pairs:
            safe = True
            for pair_ in pairs:
                if len(pair.intersection(pair_)) == 1:
                    safe = False
                    unsafe_node = list(pair.intersection(pair_))[0]
            if safe:
                pairl = list(pair)
                safe_mappings[pairl[0]] = pairl[1]
                safe_mappings[pairl[1]] = pairl[0]
            else:
                unsafe_pairs.append(pair)
                unsafe_nodes.append(unsafe_node)

        for i, g in enumerate(parent1):
            if g in safe_mappings.keys():
                child1[i] = safe_mappings[g]
        for i, g in enumerate(child2):
            if g in safe_mappings.keys():
                child2[i] = safe_mappings[g]
        for child,parent in [(child1,parent1), (child2,parent2)]:
            for i, node in enumerate(parent1):
                if node in unsafe_nodes:
                    pair1, pair2 = [pair for pair in unsafe_pairs if node in pair]
                    if len(pair1 - set(child)) > 0:
                        child[i] = list(pair1 - {node})[0]
                    elif len(pair1 - set(child)) > 0:
                        child[i] = list(pair2 - {node})[0]
                    else:
                        print('uh oh')
        return child1, child2

    def temp(self, n1, n2):
        return n1['name'] == n2['name']

    def add_names(self, pop):
        for spec in pop:
            nx.set_edge_attributes(spec, {edge: {'name': '_'.join(edge)} for edge in list(spec.edges)})
            nx.set_node_attributes(spec, self.meta_graph.nodes)

    def prune_duplicates(self, pairing):
        spec, pop = pairing

        if 0 == pop.index(spec):
            return True
        references = pop[:pop.index(spec)]
        for ref in references:
            if nx.is_isomorphic(spec, ref, self.temp, self.temp):
                return False
        return True

    def run(self, gens, pop_size, spec_opt=None):
        # make a population
        self.build_lists()
        pop = [self.build_matching() for i in range(pop_size)]
        init_pop = copy.deepcopy(pop)
        # with Pool(processes=5) as pool:
        #    pop = pool.map(self.build_matching, [None] * pop_size)
        self.add_names(pop)
        mask = [self.prune_duplicates((spec, pop)) for spec in pop]
        pop = [spec for spec in pop if mask[pop.index(spec)]]
        scores = [self.euc_fitness(complete_graph(score,self.meta_graph)) for score in pop]

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
            for i in range(pop_size // 2):
                ind1 = np.random.choice(list(range(len(pop))), p=scores)
                ind2 = np.random.choice(list(range(len(pop))), p=scores)
                gparent1 = pop[ind1].copy()
                gparent2 = pop[ind2].copy()
                while nx.is_isomorphic(gparent1, gparent2, self.temp, self.temp):
                    gparent2 = pop[np.random.choice(list(range(len(pop))), p=scores)].copy()
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
            pop = selected
            #check for duplicates
            if not (spec_opt is None):
                # start = time.time()
                 with Pool(processes=8) as pool:
                    pop = pool.map(spec_opt, pop)
                # print("threading",time.time()-start)
                #pop = [spec_opt(spec) for spec in pop]
            for spec in pop:
                if len(self.graph_to_list(spec)) != pd.Series(self.graph_to_list(spec)).nunique():
                    print(spec)
            self.add_names(pop)
            mask = [self.prune_duplicates((spec, pop)) for spec in pop]
            pop = [spec for spec in pop if mask[pop.index(spec)]]

            scores = [self.euc_fitness(complete_graph(graph, self.meta_graph)) for graph in pop]
            scores = np.array(scores) / sum(scores)
            if max(scores) > best_score:
                best_graph = complete_graph(pop[np.argmax(scores)], self.meta_graph)
                best_score = max(scores)
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
