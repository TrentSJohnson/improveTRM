import pulp


class LP:
    def euc_dis(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5

    def edge_cost(self, edge, cwgraph):
        return self.euc_dis(cwgraph.nodes[edge[2]]['x'], cwgraph.nodes[edge[2]]['y'], cwgraph.nodes[edge[0]]['xe'],
                            cwgraph.nodes[edge[0]]['ye']) + self.euc_dis(cwgraph.nodes[edge[1]]['x'],
                                                                         cwgraph.nodes[edge[1]]['y'],
                                                                         cwgraph.nodes[edge[2]]['x'],
                                                                         cwgraph.nodes[edge[2]]['y']) + self.euc_dis(
            cwgraph.nodes[edge[1]]['x'], cwgraph.nodes[edge[1]]['y'], cwgraph.nodes[edge[0]]['xs'],
            cwgraph.nodes[edge[0]]['ys'])

    def build_lists(self, meta_graph):
        overflow = [node for node in meta_graph.nodes() if meta_graph.nodes[node]['type'] == 'overflow']
        workers = [node for node in meta_graph.nodes() if meta_graph.nodes[node]['type'] == 'worker']
        underflow = [node for node in meta_graph.nodes() if meta_graph.nodes[node]['type'] == 'underflow']
        return worker, overflow, underflow

    def total_cost(self, matching, cwgraph):
        return sum([self.edge_cost(edge, cwgraph) for edge in matching])

    def solve(self, cwgraph, worker, overflow, underflow):
        # define arrangements
        possible_edges = [(w, o, u) for w in worker for o in overflow for u in underflow]

        # make problem
        wap_model = pulp.LpProblem("WAP_Model", pulp.LpMinimize)

        # make dict of them with bounds for integer assignments
        x = pulp.LpVariable.dicts(
            "edge_fractions", possible_edges, lowBound=0, upBound=1, cat=pulp.LpInteger
        )

        # create objective function
        wap_model += pulp.lpSum([x[edge] * self.edge_cost(edge, cwgraph) for edge in possible_edges])

        # contrain weights to sum to 1
        for vertex in list(workers):
            wap_model += (
                pulp.lpSum([x[edge] for edge in possible_edges if vertex in edge]) == 1,
                "assignment_bound_%s" % str(vertex),
            )

        for vertex in overflow:
            wap_model += (
                pulp.lpSum([x[edge] for edge in possible_edges if vertex in edge]) <= 1,
                "assignment_bound_%s" % str(vertex),
            )
        for vertex in underflow:
            wap_model += (
                pulp.lpSum([x[edge] for edge in possible_edges if vertex in edge]) <= 1,
                "assignment_bound_%s" % str(vertex),
            )
        wap_model.solve()
        # print("Status:", pulp.LpStatus[wap_model.status])

        matching = [edge for edge in possible_edges if x[edge].value() > 0]
        return matching, self.total_cost(matching, cwgraph)

    def test(self, graph):
        worker, overflow, underflow = self.build_lists(graph)
        return self.solve(graph, worker, overflow, underflow)
