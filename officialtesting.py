from datetime import timedelta, datetime
from tqdm import tqdm
from graphprocessing import *
from models.hs import RSL
from models.local_ratio import Local_Ratio
from models.trm import TRM
from models.uga import UGA_RSL


class Testing:
    def test(self, algorithms, data, interval, start_time_='started_at', epsilon=.1, radius=500):
        self.scores = []
        start_time_ = data[start_time][np.random.choice(data.index)]
        td = timedelta(minutes=interval)
        # epochs = int((max(data.starttime)-start_time).total_seconds()/(60*15))
        self.runtimes = []
        self.graphs = []
        self.times = []
        for i in tqdm(range(10)):
            graph = build_cwgraph(data, start_time, start_time + td, radius=radius, epsilon=epsilon)
            while len(graph.nodes) < 10:
                start_time_ = data.starttime[np.random.choice(data.index)]
                graph = build_cwgraph(data, start_time_, start_time + td)

            temp = []
            tempt = []
            for algo in algorithms:
                print(algo)
                start = datetime.now()
                temp.append(algo.test(graph)[1])
                tempt.append((datetime.now() - start).total_seconds())
            self.scores.append(temp)
            self.runtimes.append(tempt)
            start_time_ = data.starttime[np.random.choice(data.index)]
            self.graphs.append(graph)
            self.times.append(start_time)
            if i % 10 == 0:
                print(temp)
                print(tempt)
        return self.scores


if __name__ == '__main__':
    start_time = 'started_at'
    end_time = 'ended_at'
    start_station_name = 'start_station_name'
    end_station_name = 'end_station_name'

    data = pd.read_csv('data/202105-citibike-tripdata.csv')
    data = data.loc[[type(i) == str for i in data[start_station_name]]]
    data = data.loc[[type(i) == str for i in data[end_station_name]]]
    data[start_time] = data[start_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    data[end_time] = data[end_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))


    # graph = build_station_graph(data,data.loc[0,start_time],data.loc[0,start_time]+timedelta(minutes=15))
    # print(graph['E 6 St & Avenue D'])
    # overflow = {node:{} for node in graph.nodes() if graph.nodes[node]['type']=='overflow'}
    # underflow = {node:{} for node in graph.nodes() if graph.nodes[node]['type']=='underflow'}
    # print('edgesss',graph.edges)
    # print('vertixxx',graph.nodes)
    #

    filtered_data = data[(data[start_time] >= data.loc[0, start_time]) & (
                data[start_time] <= data.loc[0, start_time] + timedelta(minutes=15))]

    # cgraph = cloned_station_vertices(graph)

    cwgraph = build_cwgraph(data, data.loc[0, start_time], data.loc[0, start_time] + timedelta(minutes=15), 0.1, 500)
    # print(cwgraph.nodes)
    # verflow = {node:{} for node in graph.nodes() if cwgraph.nodes[node]['type']=='overflow'}
    # underflow = {node:{} for node in graph.nodes() if cwgraph.nodes[node]['type']=='underflow'}
    worker = {node: {} for node in cwgraph.nodes() if cwgraph.nodes[node]['type'] == 'worker'}
    print(len(worker))
    print(len(list(cwgraph.edges)))

    testing = Testing()
    scores = testing.test([UGA_RSL(), TRM(), RSL(), Local_Ratio()], data, 15, cwgraph)
    print(scores)
