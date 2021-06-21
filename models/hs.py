from networkx.algorithms import bipartite
import networkx as nx
from random import shuffle


class HS:
    def get_weight(self, graph, node1, node2, constrain_type, cwgraph):
        nodes = node1.split('|') + [node2]
        overflow = [node for node in nodes if node in list(cwgraph) and cwgraph.nodes[node]['type'] == 'overflow'][0]
        underflow = [node for node in nodes if node in list(cwgraph) and cwgraph.nodes[node]['type'] == 'underflow'][0]
        worker = list(set(nodes) - {overflow, underflow})[0]
        return self.euc_dis(cwgraph.nodes[underflow]['x'], cwgraph.nodes[underflow]['y'], cwgraph.nodes[worker]['xe'],
                            cwgraph.nodes[worker]['ye']) + self.euc_dis(cwgraph.nodes[overflow]['x'],
                                                                        cwgraph.nodes[overflow]['y'],
                                                                        cwgraph.nodes[underflow]['x'],
                                                                        cwgraph.nodes[underflow]['y']) + self.euc_dis(
            cwgraph.nodes[overflow]['x'], cwgraph.nodes[overflow]['y'], cwgraph.nodes[worker]['xs'],
            cwgraph.nodes[worker]['ys'])

    def get_type(self, cwgraph):
        overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
        underflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
        worker = [str(node) for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
        return worker, overflow, underflow

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
                        fixed.add_node(str(underflow_) + '|' + str(overflow_), bipartite=0)
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
                    fixed.add_edge(node1, node2, weight=self.get_weight(graph, node1, node2, constrain_type, cwgraph))
        # print('edges out', len(fixed.edges))
        # draw_bipartite(fixed)

        return fixed

    def update_graph(self, matching):
        g = nx.Graph()
        for v1, v2 in zip(matching.keys(), matching.values()):
            if not g.has_edge(v1, v2):
                vertices = v1.split('|') if '|' in v1 else [v1]
                vertices += v2.split('|') if '|' in v2 else [v2]
                g.add_nodes_from(vertices)
                g.add_edges_from([(v1_, v2_) for v1_ in vertices for v2_ in vertices if v1_ != v2_])
        # rint('edges',g.edges)
        return g

    def euc_dis(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5

    def euc_tri(self, worker, underflow, overflow, graph):
        worker_data = graph.nodes[worker]
        underflow_data = graph.nodes[underflow]
        overflow_data = graph.nodes[overflow]
        return self.euc_dis(worker_data['xe'], worker_data['ye'], underflow_data['x'],
                            underflow_data['y']) + self.euc_dis(worker_data['xs'], worker_data['ys'],
                                                                overflow_data['x'], overflow_data['y']) + self.euc_dis(
            underflow_data['x'], underflow_data['y'], overflow_data['x'], overflow_data['y'])

    def matching_score(self, matching, graph):
        triplets = []
        finished = set()
        temp = {}
        for v in matching.nodes:
            neighbors = list(matching.neighbors(v))
            finished = finished.union({v}).union(set(neighbors))

            temp[graph.nodes[v]['type']] = v
            temp[graph.nodes[neighbors[0]]['type']] = neighbors[0]
            temp[graph.nodes[neighbors[1]]['type']] = neighbors[1]
            triplets.append(temp)
        w = 0
        for i in triplets:
            w += self.euc_tri(i['worker'], i['underflow'], i['overflow'], graph)
        return w

    def search(self, graph, cwgraph):
        best_score = float('inf')
        best_matching = None
        stalled_rounds = 0
        constrain_types = ['worker_overflow', 'overflow_underflow', 'underflow_worker']
        i = 0
        print('searching')
        while stalled_rounds < 3:
            shuffle(constrain_types)
            for constrain_type in constrain_types:
                fixed = self.constrain_graph(graph, constrain_type, cwgraph)
                # print(fixed.edges)
                matching = bipartite.matching.minimum_weight_full_matching(fixed)
                graph = self.update_graph(matching)
            score = self.matching_score(graph, cwgraph)
            i+=1
            print(i)
            if score < best_score:
                best_score = score
                best_matching = graph
                stalled_rounds = 0
            else:
                stalled_rounds += 1
        return best_matching, best_score


class RSL:
    def make_graph(self, cwgraph):
        overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
        underflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
        worker = [str(node) for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
        shuffle(overflow)
        shuffle(underflow)
        shuffle(worker)
        full = min([len(overflow), len(underflow), len(worker)])
        overflow = overflow[:full]
        underflow = underflow[:full]
        worker = worker[:full]

        wou_triplets = [(w, o, u) for w, o, u in zip(worker, overflow, underflow)]
        graph = nx.Graph()

        for w, o, u in wou_triplets:
            graph.add_nodes_from([w, o, u])
            graph.add_edge(w, u)
            graph.add_edge(u, o)
            graph.add_edge(w, o)
        # print(graph.edges)
        # print(len(graph.edges))
        return graph

    def optimize(self, cwgraph, graph=None):
        if graph == None:
            graph = self.make_graph(cwgraph)
        hs = HS()
        return hs.search(graph, cwgraph)

    def test(self, cwgraph):
        return self.optimize(cwgraph)
