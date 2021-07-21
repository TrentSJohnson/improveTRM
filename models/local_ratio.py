import copy

import networkx as nx
import pulp

from graphprocessing import get_entities, edge_cost, score_graph, complete_graph


class Local_Ratio:

    def euc_dis(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5

    def neighborhood_weight(self, edge, possible_edges, x):
        return sum([x[edge_].value() for edge_ in possible_edges if len(set(edge).intersection(set(edge_))) > 0])

    def local_ratio(self, F, w):
        F = [edge for edge in F if w[str(edge)] > 0]
        if len(F) == 0:
            return F
        edge = F[0]

        w1 = {str(edge_): (w[str(edge_)] if len(set(edge).intersection(set(edge_))) > 0 else 0) for edge_ in F}
        w2 = {str(edge_): w[str(edge_)] - w1[str(edge_)] for edge_ in F}
        m = self.local_ratio(F, w2)
        m_ = set(m).union({edge})
        for e1 in list(m_):
            for e2 in list(m_):
                if e1 != e2 and len(set(e1).intersection(set(e2))) > 0:
                    return m
        return m_

    def matching_to_graph(self, matching, cwgraph):
        G = nx.Graph()
        for w, o, u in matching:
            G.add_node(w, **(cwgraph.nodes[w]))
            G.add_node(o, **(cwgraph.nodes[o]))
            G.add_node(u, **(cwgraph.nodes[u]))
            G.add_edge(w, u)
            G.add_edge(w, o)
            G.add_edge(u, o)
        return G

    def solve(self, cwgraph):
        workers, overflows, underflows = get_entities(cwgraph)
        # define arrangements
        possible_edges = [(w, o, u) for w in workers for o in overflows for u in underflows]

        # make problem
        wap_model = pulp.LpProblem("WAP_Model", pulp.LpMaximize)

        # make dict of them with bounds for fractional assignments
        x = pulp.LpVariable.dicts(
            "edge_fractions", possible_edges, lowBound=0
        )
        lc = 10000000
        # create objective function
        wap_model += pulp.lpSum([x[edge] * (lc - edge_cost(*edge, cwgraph)) for edge in possible_edges])

        # contrain weights to sum to 1
        for vertex in list(cwgraph.nodes):
            wap_model += (
                pulp.lpSum([x[edge] for edge in possible_edges if vertex in edge]) <= 1,
                "assignment_bound_%s" % str(vertex),
            )
        wap_model.solve()

        non_zero = [edge for edge in possible_edges if x[edge].value() > 0]
        f = []
        possible_edges_ = sorted(copy.deepcopy(possible_edges), key=lambda e: self.neighborhood_weight(e, non_zero, x))
        num_edges = len(possible_edges_)
        for e in range(num_edges):
            i = 0
            edge = possible_edges_[i]
            while self.neighborhood_weight(edge, non_zero, x) > 2:
                i += 1
                edge = possible_edges_[i]
            possible_edges_.pop(i)
            if edge in non_zero:
                non_zero.remove(edge)
            f.append(edge)
        w_dict = {str(edge): x[edge].value() for edge in f}
        matching = self.local_ratio(f, w_dict)
        matching_graph = complete_graph(self.matching_to_graph(matching, cwgraph),cwgraph)
        return matching_graph, score_graph(matching_graph, cwgraph)

    def test(self, graph):
        return self.solve(graph)
