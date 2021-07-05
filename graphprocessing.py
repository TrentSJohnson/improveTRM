import copy
from random import shuffle

import matplotlib.pyplot as plt
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


def find_closest_overflow(w, u, cwgraph, stations):
    best = float('inf')
    besto = None
    for o in stations:
        temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                       cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                        cwgraph.nodes[u]['x'],
                                                        cwgraph.nodes[u]['y']) + euc_dis(
            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
            cwgraph.nodes[o]['y'])
        if temp < best:
            besto = o
            best = temp
    return besto


def find_closest_underflow(w, o, cwgraph, stations):
    best = float('inf')
    bestu = None
    for u in stations:
        temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                       cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                        cwgraph.nodes[u]['x'],
                                                        cwgraph.nodes[u]['y']) + euc_dis(
            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
            cwgraph.nodes[o]['y'])
        if temp < best:
            bestu = u
            best = temp
    return bestu


def find_closest_end(u, cwgraph, sources):
    best = float('inf')
    bestw = None
    for w in sources:
        temp = euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'], cwgraph.nodes[u]['x'], cwgraph.nodes[u]['y'])
        if temp < best:
            bestw = w
            best = temp
    return bestw


def find_closest_start(o, cwgraph, sources):
    best = float('inf')
    bestw = None
    for w in sources:
        temp = euc_dis(cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'])
        if temp < best:
            bestw = w
            best = temp
    return bestw


def scorer2(graph, cwgraph):
    graph = complete_graph(graph, cwgraph)
    overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
    worker = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
    underflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
    c = 0
    for w in worker:
        s1, s2 = graph.neighbors(w)
        u = s1 if cwgraph.nodes[s1]['type'] == 'underflow' else s2
        o = s2 if u == s1 else s1
        c += euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                     cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                      cwgraph.nodes[u]['x'], cwgraph.nodes[u]['y']) + euc_dis(
            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'])
    return c


def complete_graph(graph, cwgraph):
    c = 0
    overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
    worker = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
    underflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'underflow']
    graph.add_nodes_from([node for node in cwgraph.nodes if not graph.has_node(node)])
    overflow_nw = [o for o in overflow if all([cwgraph.nodes[n]['type'] != 'worker' for n in graph.neighbors(o)])]
    underflow_nw = [u for u in underflow if all([cwgraph.nodes[n]['type'] != 'worker' for n in graph.neighbors(u)])]
    worker_nu = [w for w in worker if all([cwgraph.nodes[n]['type'] != 'underflow' for n in graph.neighbors(w)])]
    worker_no = [w for w in worker if all([cwgraph.nodes[n]['type'] != 'overflow' for n in graph.neighbors(w)])]

    for u in underflow_nw:
        print('STATION NOT ASSIGNED')
        w = find_closest_end(u, cwgraph, worker_nu)

        if not w is None:
            worker_nu.remove(w)
            graph.add_edge(w, u)
    for o in overflow_nw:
        print('STATION NOT ASSIGNED')
        w = find_closest_start(o, cwgraph, worker_no)
        if not w is None:
            worker_no.remove(w)
            graph.add_edge(w, o)
    fs = set(worker_nu).intersection(worker_no)
    for w in fs:
        u = find_nearest_station(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'], 'underflow', cwgraph)[0]
        graph.add_edge(w, u)
        worker_nu.remove(w)
    for w in worker_no:
        o = find_closest_overflow(w, list(graph.neighbors(w))[0], cwgraph, overflow)
        graph.add_edge(w, o)

    for w in worker_nu:
        u = find_closest_underflow(w, list(graph.neighbors(w))[0], cwgraph, underflow)
        graph.add_edge(w, u)
    return graph


def scorer(graph, cwgraph):
    c = 0
    overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
    worker = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
    underflow = [node for node in cwgraph.nodes() if
                 cwgraph.nodes[node]['type'] == 'underflow']
    if len(worker) <= len(overflow) and len(worker) <= len(underflow):
        for w in worker:
            if not w in graph.nodes:
                graph.add_node(w)
            s = list(graph.neighbors(w))

            if len(s) == 2:
                u = s[0] if cwgraph.nodes[s[0]]['type'] == 'underflow' else s[1]
                o = s[1] if cwgraph.nodes[s[0]]['type'] == 'underflow' else s[0]

            elif len(s) == 1:
                if cwgraph.nodes[s[0]]['type'] == 'underflow':
                    u = s[0]
                    best = float('inf')
                    bestp = None
                    for o in overflow:
                        temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                       cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                                        cwgraph.nodes[u]['x'],
                                                                        cwgraph.nodes[u]['y']) + euc_dis(
                            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                            cwgraph.nodes[o]['y'])
                        if best > temp:
                            best = temp
                            bestp = o
                    o = bestp

                else:
                    o = s[0]
                    best = float('inf')
                    bestp = None
                    for u in underflow:
                        temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                       cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                                        cwgraph.nodes[u]['x'],
                                                                        cwgraph.nodes[u]['y']) + euc_dis(
                            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                            cwgraph.nodes[o]['y'])
                        if best > temp:
                            best = temp
                            bestp = u
                    u = bestp

            else:
                best = float('inf')
                bestu = None
                besto = None
                for u in underflow:
                    if len([n for n in graph.neighbors(u)]):
                        for o in overflow:
                            if (not any([graph.nodes[n]['type'] == 'worker' for n in graph.neighbors(o)])):

                                temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                               cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'],
                                                                                cwgraph.nodes[w]['ye'],
                                                                                cwgraph.nodes[u]['x'],
                                                                                cwgraph.nodes[u]['y']) + euc_dis(
                                    cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                                    cwgraph.nodes[o]['y'])
                                if best > temp:
                                    best = temp
                                    bestu = u
                                    besto = o
                if bestu is None:
                    u = bestu
                o = besto
                if bestu is None and not besto is None:
                    for u in underflow:
                        temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                       cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'],
                                                                        cwgraph.nodes[w]['ye'],
                                                                        cwgraph.nodes[u]['x'],
                                                                        cwgraph.nodes[u]['y']) + euc_dis(
                            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                            cwgraph.nodes[o]['y'])
                        if best > temp:
                            best = temp
                            bestu = u
                if besto is None and not bestu is None:
                    for o in overflow:
                        temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                       cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'],
                                                                        cwgraph.nodes[w]['ye'],
                                                                        cwgraph.nodes[u]['x'],
                                                                        cwgraph.nodes[u]['y']) + euc_dis(
                            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                            cwgraph.nodes[o]['y'])
                        if best > temp:
                            best = temp
                            bestu = o
                u = bestu
                o = besto

            c += euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                         cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                          cwgraph.nodes[u]['x'], cwgraph.nodes[u]['y']) + euc_dis(
                cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'])
        return c
    elif len(underflow) <= len(overflow) and len(underflow) <= len(worker):
        for u in underflow:
            if graph.has_node(u):
                s = list(graph.neighbors(u))
                if len(s) == 2:
                    w = s[0] if cwgraph.nodes[s[0]]['type'] == 'worker' else s[1]
                    o = s[1] if cwgraph.nodes[s[0]]['type'] == 'worker' else s[0]
                    c += euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                 cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                                  cwgraph.nodes[u]['x'],
                                                                  cwgraph.nodes[u]['y']) + euc_dis(
                        cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'])

                elif len(s) == 1:
                    if cwgraph.nodes[s[0]]['type'] == 'worker':
                        w = s[0]
                        best = float('inf')
                        bestp = None
                        for o in overflow:
                            temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                           cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'],
                                                                            cwgraph.nodes[w]['ye'],
                                                                            cwgraph.nodes[u]['x'],
                                                                            cwgraph.nodes[u]['y']) + euc_dis(
                                cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                                cwgraph.nodes[o]['y'])
                            if best > temp:
                                best = temp
                                bestp = o
                        o = bestp
                        c += euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                     cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                                      cwgraph.nodes[u]['x'],
                                                                      cwgraph.nodes[u]['y']) + euc_dis(
                            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                            cwgraph.nodes[o]['y'])
        return c

    for o in overflow:
        if graph.has_node(o):

            s = list(graph.neighbors(o))
            if len(s) == 2:
                w = s[0] if cwgraph.nodes[s[0]]['type'] == 'worker' else s[1]
                u = s[1] if cwgraph.nodes[s[0]]['type'] == 'worker' else s[0]
                c += euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                             cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                              cwgraph.nodes[u]['x'], cwgraph.nodes[u]['y']) + euc_dis(
                    cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'])

            elif len(s) == 1:
                if cwgraph.nodes[s[0]]['type'] == 'worker':
                    w = s[1]
                    best = float('inf')
                    bestp = None
                    for u in underflow:
                        temp = euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                       cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                                        cwgraph.nodes[u]['x'],
                                                                        cwgraph.nodes[u]['y']) + euc_dis(
                            cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                            cwgraph.nodes[o]['y'])
                        if best > temp:
                            best = temp
                            bestp = u
                    u = bestp
                    c += euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                 cwgraph.nodes[u]['y']) + euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'],
                                                                  cwgraph.nodes[u]['x'],
                                                                  cwgraph.nodes[u]['y']) + euc_dis(
                        cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'])

    return c


def build_station_graph(data, starttime, stoptime):
    graphx = nx.Graph()
    start_time = 'started_at'
    end_time = 'ended_at'
    start_station_name = 'start_station_name'
    end_station_name = 'end_station_name'
    filtered_data = data[(data[start_time] >= starttime) & (data[start_time] <= stoptime)].dropna()
    locations = list(set(filtered_data[start_station_name].values).union(filtered_data[end_station_name].values))
    vertices = pd.Series([{}] * len(locations), index=locations)

    finished = {vertex: False for vertex in vertices.index}
    print('adding vertex data')
    for i in filtered_data.index:
        start = filtered_data.loc[i, start_station_name]
        if (not finished[start]):
            finished[start] = True
            x, y, _, __ = utm.from_latlon(filtered_data.loc[i, start_lat], filtered_data.loc[i, start_lng])
            vertices[start] = {'change': 0, 'x': x, 'y': y, 'type': ''}

        end = filtered_data.loc[i, end_station_name]
        if (not finished[end]):
            finished[end] = True
            x, y, _, __ = utm.from_latlon(filtered_data.loc[i, end_lat], filtered_data.loc[i, end_lng])
            vertices[end] = {'change': 0, 'x': x, 'y': y, 'type': ''}
    for i in filtered_data[start_station_name]:
        vertices[i]['change'] -= 1
    for i in filtered_data[end_station_name]:
        vertices[i]['change'] += 1
    for loc in locations:
        if vertices[loc]['change'] > 0:
            vertices[loc]['type'] = 'overflow'
        elif vertices[loc]['change'] < 0:
            vertices[loc]['type'] = 'underflow'
        else:
            vertices[loc]['type'] = ''
    vertices = vertices.loc[[x['type'] != '' for x in vertices]]
    for vertex in vertices.index:
        graphx.add_node(vertex, bipartite=int(vertices[vertex]['type'] == 'overflow'))
    # print(vertices)
    nx.set_node_attributes(graphx, {i: vertices[i] for i in vertices.index})
    for i in range(len(vertices.index)):
        for j in range(i, len(vertices.index)):
            if graphx.nodes[vertices.index[i]]['bipartite'] != graphx.nodes[vertices.index[j]][
                'bipartite'] and not graphx.has_edge(i, j):
                x1 = vertices[vertices.index[i]]['x']
                y1 = vertices[vertices.index[i]]['y']
                x2 = vertices[vertices.index[j]]['x']
                y2 = vertices[vertices.index[j]]['y']
                graphx.add_edge(vertices.index[i], vertices.index[j], weight=euc_dis(x1, y1, x2, y2))
    return graphx


def build_cwgraph(data, starttime, stoptime, epsilon, radius):
    graph = build_station_graph(data, starttime, stoptime)
    filtered_data = data[(data[start_time] >= starttime) & (data[start_time] <= stoptime)].dropna()
    cgraph = cloned_station_vertices(graph)
    workers = []
    for i, s, e in zip(filtered_data.index, filtered_data['start_station_name'], filtered_data['end_station_name']):

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

        workers.append({'type': 'worker', 'change': 0, 'xs': xs + radius * (1 - 2 * np.random.random()),
                        'ys': ys + 500 * (1 - 2 * np.random.random()), 'xe': xe + radius * (1 - 2 * np.random.random()),
                        'ye': ye + 500 * (1 - 2 * np.random.random())})
    cwgraph = copy.deepcopy(cgraph)
    # print(cwgraph.nodes)
    shuffle(workers)
    workers = workers[:int(epsilon * len(workers))]
    cwgraph.add_nodes_from([str(i) for i in range(len(workers))])
    nx.set_node_attributes(cwgraph, {str(i): w for i, w in enumerate(workers)})
    cwgraph.add_edges_from([(str(i), s) for i in range(len(workers)) for s in cgraph.nodes])

    return cwgraph


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
                    if graphx.nodes[node]['bipartite'] != graphx.nodes[node_]['bipartite']:
                        graphx.add_edge(node_, new_node_name,
                                        weight=euc_dis(new_node_data['x'], new_node_data['y'], graphx.nodes[node_]['x'],
                                                       graphx.nodes[node_]['y']))
    return graphx


def draw_bipartite(B):
    l, r = nx.bipartite.sets(B)
    pos = {}
    pos.update((node, (1, index)) for index, node in enumerate(l))
    pos.update((node, (2, index)) for index, node in enumerate(r))
    nx.draw(B, pos=pos)
    plt.show()


def draw_graph(B):
    pos = {}
    # Update position for node from each group
    pos.update((node, (B.nodes[node]['x'], B.nodes[node]['y'])) for node in list(B.nodes))
    nx.draw(B, pos=pos)
    plt.show()


def euc_dis(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5


def find_nearest_station(x, y, stype, cwgraph):
    best_station = None
    best_score = float('inf')
    for station in cwgraph.nodes:
        if cwgraph.nodes[station]['type'] == stype and euc_dis(cwgraph.nodes[station]['x'], cwgraph.nodes[station]['y'],
                                                               x, y) < best_score:
            best_score = euc_dis(cwgraph.nodes[station]['x'], cwgraph.nodes[station]['y'], x, y)
            best_station = station
    return best_station, best_score


def get_shortest_assignment(cwgraph):
    dist = 0
    overflow = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'overflow']
    worker = [node for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker']
    underflow = [node for node in cwgraph.nodes() if
                 cwgraph.nodes[node]['type'] == 'underflow']

    for w in worker:

        best_dis = float('inf')
        for o in overflow:
            for u in underflow:
                temp_dist = 0
                temp_dist += euc_dis(cwgraph.nodes[w]['xe'], cwgraph.nodes[w]['ye'], cwgraph.nodes[u]['x'],
                                     cwgraph.nodes[u]['y'])
                temp_dist += euc_dis(cwgraph.nodes[w]['xs'], cwgraph.nodes[w]['ys'], cwgraph.nodes[o]['x'],
                                     cwgraph.nodes[o]['y'])
                temp_dist += euc_dis(cwgraph.nodes[o]['x'], cwgraph.nodes[o]['y'], cwgraph.nodes[u]['x'],
                                     cwgraph.nodes[u]['y'])
                best_dis = min([best_dis, temp_dist])
        dist += best_dis
    return dist