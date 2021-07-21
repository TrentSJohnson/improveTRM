from random import shuffle

import networkx as nx
import numpy as np
import pandas as pd
import utm

start_time = 'started_at'
end_time = 'ended_at'
start_station_name = 'start_station_name'
end_station_name = 'end_station_name'
start_lat = 'start_lat'
start_lng = 'start_lng'
end_lat = 'end_lat'
end_lng = 'end_lng'
types = ['worker', 'overflow', 'underflow']


def get_entities(cwgraph):
    overflows = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
    workers = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
    underflows = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
    return workers, overflows, underflows


def edge_cost(n1, n2, n3, cwgraph):
    """
    Finds which node is the worker overflow and underflow and finds the euc_dis of the nodes
    """
    n1t = cwgraph.nodes[n1]['type']
    n2t = cwgraph.nodes[n2]['type']
    w = n1 if n1t == 'worker' else (n2 if n2t == 'worker' else n3)
    o = n1 if n1t == 'overflow' else (n2 if n2t == 'overflow' else n3)
    u = n1 if n1t == 'underflow' else (n2 if n2t == 'underflow' else n3)
    return euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'], cwgraph.nodes[u]['x'],
                   cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'],
                                                    cwgraph.nodes[o]['x'],
                                                    cwgraph.nodes[o]['y']) + euc_dis(cwgraph.nodes[o]['x'],
                                                                                     cwgraph.nodes[o]['y'],
                                                                                     cwgraph.nodes[u]['x'],
                                                                                     cwgraph.nodes[u]['y'])


def edge2_cost(n1, n2, cwgraph):
    """
    Finds cost of 2 vertices
    """
    n1t = cwgraph.nodes[n1]['type']
    n2t = cwgraph.nodes[n2]['type']
    w = n1 if n1t == 'worker' else (n2 if n2t == 'worker' else None)
    o = n1 if n1t == 'overflow' else (n2 if n2t == 'overflow' else None)
    u = n1 if n1t == 'underflow' else (n2 if n2t == 'underflow' else None)
    if u is None:
        return euc_dis(cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                       cwgraph.nodes[o]['y'])
    if w is None:
        return euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'], cwgraph.nodes[u]['y'])
    return euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'], cwgraph.nodes[u]['x'],
                   cwgraph.nodes[u]['y'])


def find_opt_end(w, o, us, cwgraph):
    """
    Finds the optimal end station for a worker
    Args:
        w: str 
            worker node
        o: str
            overflow node
        us: list
            list of underflow nodes
        cwgraph: Graph
            complete data graph
    Returns: str    
        the optimal end node
    """
    if len(us) == 1:
        return us[0]
    dis = edge_cost(us[0], o, w, cwgraph)
    recu = find_opt_end(w, o, us[1:], cwgraph)
    recdis = edge_cost(recu, o, w, cwgraph)
    return us[0] if recdis > dis else recu


def find_opt_start(w, u, os, cwgraph):
    """
    Finds the optimal start station for a worker
    Args:
        w: str 
            worker node
        u: str
            underflow node
        os: list
            list of overflow nodes
        cwgraph: Graph
            complete data graph
    Returns: str
        the optimal start node
    """
    if len(os) == 1:
        return os[0]
    dis = edge_cost(os[0], u, w, cwgraph)
    reco = find_opt_start(w, u, os[1:], cwgraph)
    recdis = edge_cost(reco, u, w, cwgraph)
    return os[0] if recdis > dis else reco


def find_opt_stations(w, os, us, cwgraph):
    if len(os) == 1:
        return os[0], find_opt_end(w, os[0], us, cwgraph)
    u = find_opt_end(w, os[0], us, cwgraph)
    dis = edge_cost(w, os[0], u, cwgraph)
    reco, recu = find_opt_stations(w, os[1:], us, cwgraph)
    recdis = edge_cost(reco, recu, w, cwgraph)
    return (os[0], u) if recdis > dis else (reco, recu)


def check_fully_assigned(graph, cwgraph):
    workers, overflows, underflows = get_entities(cwgraph)
    if len(workers) < len(overflows):
        for w in workers:
            neighbors = set(graph.neighbors(w))
            if len(neighbors - set(overflows)) != 1 or len(neighbors - set(underflows)) != 1:
                return False
        return True
    for s in overflows + underflows:
        if len(list(cwgraph.neighbors(s))) == 0:
            return False
    return True


def score_graph(graph, cwgraph):
    workers = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
    c = 0
    for w in workers:
        if(len(list(graph.neighbors(w)))>2):
            print('f')
        s1, s2 = graph.neighbors(w)
        u = s1 if cwgraph.nodes[s1]['type'] == 'underflow' else s2
        o = s2 if u == s1 else s1
        c += edge_cost(w, u, o, cwgraph)
    return c


def partial_scorer(graph, cwgraph):
    workers = [node for node in graph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
    c = 0
    for w in workers:
        neighbors = list(graph.neighbors(w))
        if len(neighbors) == 2:
            s1, s2 = graph.neighbors(w)
            c += edge_cost(w, s1, s2, cwgraph)
        elif len(neighbors) == 1:
            c += edge2_cost(w, neighbors[0], cwgraph)
    return c


def complete_graph(graph, cwgraph):
    graph = graph.copy()
    workers, overflows, underflows = get_entities(cwgraph)

    graph.add_nodes_from([node for node in cwgraph.nodes if not graph.has_node(node)])
    worker_nu = [w for w in workers if all([cwgraph.nodes[n]['type'] != 'underflow' for n in graph.neighbors(w)])]
    worker_no = [w for w in workers if all([cwgraph.nodes[n]['type'] != 'overflow' for n in graph.neighbors(w)])]
    fs = list(set(worker_nu).intersection(worker_no))

    for w in fs:
        o, u = find_opt_stations(w, overflows, underflows, cwgraph)
        graph.add_edge(w, u)
        graph.add_edge(w, o)
        worker_nu.remove(w)
        worker_no.remove(w)
    for w in worker_no:
        o = find_opt_start(w, list(graph.neighbors(w))[0], overflows, cwgraph)
        graph.add_edge(w, o)

    for w in worker_nu:
        u = find_opt_end(w, list(graph.neighbors(w))[0], underflows, cwgraph)
        graph.add_edge(w, u)
    return graph


def build_station_graph(data, starttime, stoptime):
    graphx = nx.Graph()
    filtered_data = data[(data[start_time] >= starttime) & (data[start_time] <= stoptime)].dropna()
    dt = stoptime - starttime
    filtered_data_past = data[(data[start_time] >= starttime - dt) & (data[start_time] <= stoptime - dt)].dropna()
    locations = list(set(filtered_data[start_station_name].values).union(filtered_data[end_station_name].values))
    vertices = pd.Series([{}] * len(locations), index=locations)

    finished = {vertex: False for vertex in vertices.index}
    print('adding vertex data')
    for i in filtered_data.index:
        start = filtered_data.loc[i, start_station_name]
        if (not finished[start]):
            finished[start] = True
            x, y, _, __ = utm.from_latlon(filtered_data.loc[i, start_lat], filtered_data.loc[i, start_lng])
            vertices[start] = {'change': 0, 'x': x, 'y': y, 'type': '', 'name': start}

        end = filtered_data.loc[i, end_station_name]
        if (not finished[end]):
            finished[end] = True
            x, y, _, __ = utm.from_latlon(filtered_data.loc[i, end_lat], filtered_data.loc[i, end_lng])
            vertices[end] = {'change': 0, 'x': x, 'y': y, 'type': '', 'name': end}
    for i in filtered_data_past[start_station_name]:
        if i in vertices.index:
            vertices[i]['change'] -= 1
    for i in filtered_data_past[end_station_name]:
        if i in vertices.index:
            vertices[i]['change'] += 1
    for loc in locations:
        if vertices[loc]['change'] > 0:
            vertices[loc]['type'] = 'overflow'
        elif vertices[loc]['change'] < 0:
            vertices[loc]['type'] = 'underflow'
        else:
            vertices[loc]['type'] = ''
    vertices = vertices.loc[[x['type'] != '' for x in vertices]]
    overflows = [i for i in vertices.index if vertices[i]['type'] == 'overflow']
    underflows = [i for i in vertices.index if vertices[i]['type'] == 'underflow']
    graphx.add_nodes_from(overflows + underflows)
    nx.set_node_attributes(graphx, {i: vertices[i] for i in (overflows + underflows)})
    for o in overflows:
        for u in underflows:
            graphx.add_edge(o, u, weight=edge2_cost(o, u, graphx))
    return graphx


def build_cwgraph(data, starttime, stoptime, ratio, radius):
    """
    Builds a graph with 
    """
    graph = build_station_graph(data, starttime, stoptime)
    n_workers = int(len(graph.nodes) / 2 * ratio)
    workers = [str(i) for i in range(n_workers)]
    filtered_data = data[(data[start_time] >= starttime) & (data[start_time] <= stoptime)].dropna()
    cgraph = cloned_station_vertices(graph)
    overflow = [node for node in cgraph.nodes() if cgraph.nodes[node]['type'] == 'overflow']
    underflow = [node for node in cgraph.nodes() if cgraph.nodes[node]['type'] == 'underflow']
    worker_data = []
    while n_workers > len(worker_data):
        i = np.random.choice(filtered_data.index)
        s = filtered_data.loc[i, start_station_name]
        e = filtered_data.loc[i, end_station_name]
        if s in cgraph.nodes:
            xs = cgraph.nodes[s]['x']
            ys = cgraph.nodes[s]['y']
        else:
            xs, ys, _, __ = utm.from_latlon(filtered_data.loc[i, 'start_lat'], filtered_data.loc[i, 'start_lng'])
        if e in cgraph.nodes:
            xe = cgraph.nodes[e]['x']
            ye = cgraph.nodes[e]['y']
        else:
            xe, ye, _, __ = utm.from_latlon(filtered_data.loc[i, 'end_lat'], filtered_data.loc[i, 'end_lng'])
        worker_data.append({'type': 'worker', 'change': 0, 'xs': xs + radius * (1 - 2 * np.random.random()),
                            'ys': ys + 500 * (1 - 2 * np.random.random()),
                            'xe': xe + radius * (1 - 2 * np.random.random()),
                            'ye': ye + 500 * (1 - 2 * np.random.random()),
                            'name': str(len(worker_data))})
    cgraph.add_nodes_from(workers)
    nx.set_node_attributes(cgraph, {i: w for i, w in zip(workers, worker_data)})
    for w in workers:
        for o in overflow:
            cgraph.add_edge(w, o, weight=edge2_cost(w, o, cgraph))
        for u in underflow:
            cgraph.add_edge(w, u, weight=edge2_cost(w, u, cgraph))

    return cgraph


def cloned_station_vertices(graphx):
    nodes = list(graphx.nodes())
    for node in nodes:
        if np.abs(graphx.nodes[node]['change']) > 1:
            for i in range(np.abs(graphx.nodes[node]['change']) - 1):
                i = i + 1
                new_node_name = str(node) + str(i)
                graphx.add_node(new_node_name, **graphx.nodes[node])
                new_node_data = graphx.nodes[new_node_name]
                for node_ in graphx.nodes:
                    if graphx.nodes[node]['type'] != graphx.nodes[node_]['type']:
                        graphx.add_edge(node_, new_node_name, **new_node_data)
    workers, overflows, underflows = get_entities(graphx)
    m = min((len(overflows), len(underflows)))
    shuffle(overflows)
    shuffle(underflows)
    overflows_restr = overflows[:m]
    underflows_restr = underflows[:m]
    for o in overflows:
        if not (o in overflows_restr):
            graphx.remove_node(o)
    for u in underflows:
        if not (u in underflows_restr):
            graphx.remove_node(u)
    return graphx


def euc_dis(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5


def get_shortest_assignment(cwgraph):
    dist = 0
    workers, overflows, underflows = get_entities(cwgraph)
    for w in workers:
        o, u = find_opt_stations(w, overflows, underflows, cwgraph)
        dist += edge_cost(w, o, u, cwgraph)
    return dist
