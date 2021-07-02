import networkx as nx
from networkx.algorithms import bipartite


class TRM:
    def euc_tri(self, worker, underflow, overflow, graph):
        worker_data = graph.nodes[worker]
        underflow_data = graph.nodes[underflow]
        overflow_data = graph.nodes[overflow]
        return self.euc_dis(worker_data['xe'], worker_data['ye'], underflow_data['x'],
                            underflow_data['y']) + self.euc_dis(worker_data['xs'], worker_data['ys'],
                                                                overflow_data['x'], overflow_data['y']) + self.euc_dis(
            underflow_data['x'], underflow_data['y'], overflow_data['x'], overflow_data['y'])

    def matching_to_triplets(self, matching, graph):
        triplets = []
        finished = []
        for u1 in matching.keys():
            temp = {}
            if '|' in u1:
                u2 = matching[u1]
                u1, u3 = u1.split('|')

                if not (u1 in finished):
                    temp[graph.nodes[u1]['type']] = u1
                    temp[graph.nodes[u2]['type']] = u2
                    temp[graph.nodes[u3]['type']] = u3
                    triplets.append(temp)
                    finished += [u1, u2, u3]
        return triplets

    def matching_score(self, matching, graph):
        triplets = self.matching_to_triplets(matching, graph)
        w = 0
        for i in triplets:
            w += self.euc_tri(i['worker'], i['underflow'], i['overflow'], graph)
        return w

    def euc_dis(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5

    def matching_to_graph(self, matching, graph):
        new_graph = nx.Graph()
        triplets = self.matching_to_triplets(matching, graph)
        for triplet in triplets:
            v = triplet.values()
            new_graph.add_nodes_from(v)
            new_graph.add_edges_from([(v1, v2) for v1 in v for v2 in v if v1 != v2])
        return new_graph

    def solve(self, cgraph, cwgraph, worker):
        matching = bipartite.matching.minimum_weight_full_matching(cgraph)
        spwgraph = nx.Graph()
        for key in matching.keys():
            if cgraph.nodes[key]['type'] == 'overflow':
                # print('is overflow')
                for w, worker_ in enumerate(worker):
                    spwgraph.add_edge((key + '|' + matching[key]), worker_,
                                      weight=self.euc_dis(cwgraph.nodes[key]['x'], cwgraph.nodes[key]['y'],
                                                          cgraph.nodes[matching[key]]['x'],
                                                          cgraph.nodes[matching[key]]['y']) +
                                             self.euc_dis(cwgraph.nodes[worker_]['xs'], cwgraph.nodes[worker_]['ys'],
                                                          cgraph.nodes[key]['x'], cgraph.nodes[key]['y']) +
                                             self.euc_dis(cwgraph.nodes[worker_]['xe'], cwgraph.nodes[worker_]['ye'],
                                                          cgraph.nodes[matching[key]]['x'],
                                                          cgraph.nodes[matching[key]]['y'])
                                      )
        # print('spw',spwgraph.edges)
        matching2 = bipartite.matching.minimum_weight_full_matching(spwgraph)
        # print(matching2)
        return self.matching_to_graph(matching2, cwgraph), self.matching_score(matching2, cwgraph)

    def test(self, cwgraph):
        cgraph = nx.Graph()
        cgraph.add_nodes_from([node for node in cwgraph.nodes if cwgraph.nodes[node]['type'] != 'worker'])
        nx.set_node_attributes(cgraph, {i: cwgraph.nodes[i] for i in cgraph.nodes})

        worker = list(set(cwgraph.nodes) - set(cgraph.nodes))
        for edge in cwgraph.edges:
            if cgraph.has_node(edge[0]) and cgraph.has_node(edge[1]):
                cgraph.add_edge(edge[0], edge[1],
                                weight=self.euc_dis(cwgraph.nodes[edge[0]]['x'], cwgraph.nodes[edge[0]]['y'],
                                                    cwgraph.nodes[edge[1]]['x'], cwgraph.nodes[edge[1]]['y']))
        # print('nodedata',cgraph.nodes[list(cgraph.nodes)[0]])
        # worker = [node for node in cwgraph.nodes if cwgraph.nodes[node]['type']=='worker']
        # print(len(worker))

        return self.solve(cgraph, cwgraph, worker)
