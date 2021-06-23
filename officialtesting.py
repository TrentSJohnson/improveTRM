from datetime import timedelta, datetime
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from graphprocessing import *
from models.hs import RSL
from models.local_ratio import Local_Ratio
from models.trm import TRM
from models.uga import UGA_RSL, UGA


class Testing:
    def test_(self, algorithms, data, interval, start_time_, epsilon, radius):
        print('started thread')
        start_time_dt = data[start_time_][np.random.choice(data.index)]
        td = timedelta(minutes=interval)
        graph = build_cwgraph(data, start_time_dt, start_time_dt + td, radius=radius, epsilon=epsilon)
        while len(graph.nodes) < 4:
            print('failed')
            start_time_dt = data[start_time_][np.random.choice(data.index)]
            graph = build_cwgraph(data, start_time_dt, start_time_dt + td, radius=radius, epsilon=epsilon)
        print('nodes:', len(graph.nodes))
        temp = []
        tempt = []
        for algo in algorithms:
            print(str(algo))
            start = datetime.now()
            score = algo.test(graph)[1]
            temp.append(score)
            tempt.append((datetime.now() - start).total_seconds())
            print(score)
        self.scores.append(temp)
        self.runtimes.append(tempt)
        self.graphs.append(graph)
        self.times.append(start_time_dt)
        return (temp,tempt,graph,start_time_dt)

    def test(self, algorithms, data, interval, start_time_='started_at', epsilon=1, radius=500):
        self.scores = []
        # epochs = int((max(data.starttime)-start_time).total_seconds()/(60*15))
        self.runtimes = []
        self.graphs = []
        self.times = []
        with Pool() as pool:
            print('threading')
            data = pool.starmap(self.test_, [(algorithms, data, interval, start_time_, epsilon, radius) for i in range(5)])
        scores = [data[i][0] for i in range(len(data))]
        runtimes = [data[i][1] for i in range(len(data))]
        graphs = [data[i][2] for i in range(len(data))]
        times = [data[i][3] for i in range(len(data))]

        return scores, runtimes, times, graphs


if __name__ == '__main__':
    start_time = 'started_at'
    end_time = 'ended_at'
    start_station_name = 'start_station_name'
    end_station_name = 'end_station_name'

    data = pd.read_csv('data/202105-citibike-tripdata.csv')
    data = data.loc[[type(i) == str for i in data[start_station_name]]]
    data = data.loc[[type(i) == str for i in data[end_station_name]]]
    #data = data.loc[data.index[:10000]]
    data[start_time] = data[start_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    data[end_time] = data[end_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    # graph = build_station_graph(data,data.loc[0,start_time],data.loc[0,start_time]+timedelta(minutes=15))
    # print(graph['E 6 St & Avenue D'])
    # overflow = {node:{} for node in graph.nodes() if graph.nodes[node]['type']=='overflow'}
    # underflow = {node:{} for node in graph.nodes() if graph.nodes[node]['type']=='underflow'}
    # print('edgesss',graph.edges)
    # print('vertixxx',graph.nodes)
    #

    # cgraph = cloned_station_vertices(graph)

    cwgraph = build_cwgraph(data, data.loc[0, start_time], data.loc[0, start_time] + timedelta(minutes=1), 1, 500)
    # print(cwgraph.nodes)
    # verflow = {node:{} for node in graph.nodes() if cwgraph.nodes[node]['type']=='overflow'}
    # underflow = {node:{} for node in graph.nodes() if cwgraph.nodes[node]['type']=='underflow'}
    worker = {node: {} for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker'}
    print(len(worker))
    print(len(list(cwgraph.edges)))

    testing = Testing()
    scores, runtimes, times, graphs = testing.test([UGA(), Local_Ratio(), TRM(), RSL(), UGA_RSL()], data, 10)

    scores_df = pd.DataFrame(np.abs(scores), columns=['UGA', 'Local_Ratio', 'TRM', 'RSL', 'UGA_RSL'])
    runtimes_df = pd.DataFrame(np.abs(runtimes), columns=['UGA', 'Local_Ratio', 'TRM', 'RSL', 'UGA_RSL'])
    scores_df.to_csv('outputs/scores.csv')
    runtimes_df.to_csv('outputs/times.csv')
    times_df = pd.DataFrame(times)
    graphs_df = pd.DataFrame(graphs)
    times_df.to_csv('outputs/times.csv')
    graphs_df.to_csv('outputs/graphs.csv')
    for g, graph in enumerate(graphs):
        nx.write_adjlist(graph, "outputs/test" + str(g) + ".adjlist")
    print(scores_df.head())
    print(runtimes_df.head())
