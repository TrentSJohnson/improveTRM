import networkx as nx
from networkx.algorithms import bipartite

from graphprocessing import edge2_cost, edge_cost, get_entities, score_graph, complete_graph


class TRM:

    def decontr_matching(self, contr_matching, cwgraph):
        g = nx.Graph()
        for entity in get_entities(cwgraph):
            g.add_nodes_from(entity)
        for v1 in contr_matching.keys():
            v2 = contr_matching[v1]
            if not g.has_edge(v1, v2):
                vertices = v1.split('|') if '|' in v1 else [v1]
                vertices += v2.split('|') if '|' in v2 else [v2]
                g.add_edges_from([(v1_, v2_) for v1_ in vertices for v2_ in vertices if v1_ != v2_])
        return g

    def solve(self, cgraph, cwgraph):
        workers = [node for node in cwgraph.nodes if cwgraph.nodes[node]['type'] == 'worker']
        matching = bipartite.matching.minimum_weight_full_matching(cgraph)
        spwgraph = nx.Graph()
        for key in matching.keys():
            if cgraph.nodes[key]['type'] == 'overflow':
                for worker_ in workers:
                    spwgraph.add_edge((key + '|' + matching[key]), worker_,
                                      weight=edge_cost(worker_, key, matching[key], cwgraph))

        matching2 = bipartite.matching.minimum_weight_full_matching(spwgraph)
        self.graph = self.decontr_matching(matching2, cwgraph)
        self.graph = complete_graph(self.graph, cwgraph)
        self.score = score_graph(self.graph, cwgraph)
        return self.graph, self.score

    def test(self, cwgraph):
        cgraph = nx.Graph()
        cgraph.add_nodes_from([node for node in cwgraph.nodes if cwgraph.nodes[node]['type'] != 'worker'])
        nx.set_node_attributes(cgraph, {i: cwgraph.nodes[i] for i in cgraph.nodes})

        for edge in cwgraph.edges:
            if cgraph.has_node(edge[0]) and cgraph.has_node(edge[1]):
                cgraph.add_edge(edge[0], edge[1], weight=edge2_cost(*edge, cwgraph))
        return self.solve(cgraph, cwgraph)

    def get_result(self):
        return self.graph, self.score
