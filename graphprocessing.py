import copy
from random import shuffle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import utm


def build_station_graph(data, starttime, stoptime):
    graphx = nx.Graph()
    start_time='started_at'
    end_time='ended_at'
    start_station_name='start_station_name'
    end_station_name='end_station_name'
    filtered_data = data[(data[start_time] >= starttime) & (data[start_time] <= stoptime)]
    locations = list(set(filtered_data[start_station_name].values).union(filtered_data[end_station_name].values))
    vertices = pd.Series([{}] * len(locations), index=locations)

    finished = {vertex: False for vertex in vertices.index}
    print('adding vertex data')
    for i in filtered_data.index:
        start = filtered_data.loc[i, start_station_name]
        if (not finished[start]):
            finished[start] = True
            x, y, _, __ = utm.from_latlon(filtered_data.loc[i, 'start_lat'], filtered_data.loc[i, 'start_lng'])
            vertices[start] = {'change': 0, 'x': x, 'y': y, 'type': ''}

        end = filtered_data.loc[i, end_station_name]
        if (not finished[end]):
            finished[end] = True
            x, y, _, __ = utm.from_latlon(filtered_data.loc[i, 'end_lat'], filtered_data.loc[i, 'end_lng'])
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
    start_time = 'started_at'
    end_time = 'ended_at'
    start_station_name = 'start_station_name'
    end_station_name = 'end_station_name'
    graph = build_station_graph(data, starttime, stoptime)
    filtered_data = data[(data[start_time] >= starttime) & (data[start_time] <= stoptime)]
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
                        'ys': ys+500 * (1 - 2 * np.random.random()), 'xe': xe + radius * (1 - 2 * np.random.random()),
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

    # Updat
    # e position for node from each group
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
