from datetime import timedelta, datetime

import numpy as np

from graphprocessing import *
from models.hs import RLS, TRM_RLS
from models.local_ratio import Local_Ratio
from models.trm import TRM
from models.uga import UGA_RSL


class Testing:
    def test_(self, algorithms, data, interval, epsilon, radius, begin_interval, end_interval):
        print('started thread')
        filtered_data = data

        if not begin_interval is None:
            filtered_data = filtered_data[(filtered_data[start_time] >= begin_interval)].dropna()
        if not end_interval is None:
            filtered_data = filtered_data[(filtered_data[start_time] >= begin_interval)].dropna()
        if len(filtered_data[start_time]) <10:
            return None
        start_time_dt = filtered_data[start_time][np.random.choice(filtered_data.index)]

        td = timedelta(minutes=interval)
        cwgraph = build_cwgraph(data, start_time_dt, start_time_dt + td, radius=radius, epsilon=epsilon)
        while len(cwgraph.nodes) < 4:
            print('failed')
            start_time_dt = data[start_time][np.random.choice(data.index)]
            cwgraph = build_cwgraph(data, start_time_dt, start_time_dt + td, radius=radius, epsilon=epsilon)
        print('nodes:', len(cwgraph.nodes))
        temp = []
        tempt = []
        short = get_shortest_assignment(cwgraph)
        print('shortest:',short)
        for algo in algorithms:
            if len(list(cwgraph.nodes)) < 500 or (algo != algorithms[1]):
                print(str(algo))
                start = datetime.now()
                graph = algo.test(cwgraph)[0]
                score = scorer2(graph, cwgraph=cwgraph)
                temp.append(score)
                tempt.append((datetime.now() - start).total_seconds())
                print(score)
            else:
                temp.append(np.nan)
                tempt.append(np.nan)
        return (temp, tempt, len(list(cwgraph.nodes)), start_time_dt, short)

    def test(self, algorithms, data, interval, epsilon=1.0, radius=500, trials=1, begin_interval=None,
             end_interval=None):
        self.scores = []
        # epochs = int((max(data.starttime)-start_time).total_seconds()/(60*15))
        self.runtimes = []
        self.graphs = []
        self.times = []
        results = []
        for i in range(trials):
            r = self.test_(algorithms, data, interval, epsilon, radius, begin_interval, end_interval)
            if not r is None:
                results.append(r)
        scores = [results[i][0] for i in range(len(results))]
        runtimes = [results[i][1] for i in range(len(results))]
        graph_sizes = [results[i][2] for i in range(len(results))]
        times = [results[i][3] for i in range(len(results))]
        sdists = [results[i][4] for i in range(len(results))]
        return scores, runtimes, times, graph_sizes, sdists


def run_test(data, name='citi', interval=1, begin_interval=None, end_interval=None, trial=''):
    testing = Testing()
    scores, runtimes, times, graphs, sdists = testing.test([TRM(), Local_Ratio(), UGA_RSL(), RLS(), TRM_RLS()],
                                                           data, interval=interval, epsilon=.2, trials=1,
                                                           begin_interval=begin_interval, end_interval=end_interval)

    cols = ['TRM', 'Local Ratio', 'GHS', 'RHS', 'TRHS']
    scores_df = pd.DataFrame(np.abs(scores),
                             columns=cols)
    runtimes_df = pd.DataFrame(np.abs(runtimes),
                               columns=cols)
    scores_df.to_csv('outputs/scores_' + trial + name + '.csv')
    runtimes_df.to_csv('outputs/runtimes_' + trial + name + '.csv')
    times_df = pd.DataFrame(times)
    graphs_df = pd.DataFrame(graphs)
    times_df.to_csv('outputs/times_' + trial + name + '.csv')
    graphs_df.to_csv('outputs/graphs_' + trial + name + '.csv')
    sdists_df = pd.DataFrame(sdists)
    sdists_df.to_csv('outputs/sdists_' + trial + name + '.csv')


def metrodate(s):
    date, time = s.split(' ')
    month, day, year = date.split('/')

    hour, minute = time.split(':')
    return {'month': int(month), 'day': int(day), 'year': int(year), 'hour': int(hour), 'minute': int(minute)}


if __name__ == '__main__':
    """
    metro = pd.read_csv('data/metro-trips-2021-q1.csv')
    metro[start_time] = metro['start_time'].apply(lambda x: datetime(**metrodate(x)))
    metro[end_time] = metro['end_time'].apply(lambda x: datetime(**metrodate(x)))
    metro[start_station_name] = metro['start_station'].apply(str)
    metro[end_station_name] = metro['end_station'].apply(str)
    metro[start_lat] = metro['start_lat']
    metro[start_lng] = metro['start_lon']
    metro[end_lng] = metro['end_lon']
    metro[end_lng] = metro['end_lon']
   
    for i in range(28):
        if start.weekday() in [0, 6]:
            run_test(metro, name='metro', interval=15, begin_interval=start, end_interval=start+timedelta(hours=10), trial='weekend'+str(i))
        else:
            run_test(metro, name='metro', interval=15, begin_interval=start, end_interval=start+timedelta(hours=10), trial='weekday' + str(i))
        start += timedelta(days=1)
        """
    # graph = build_station_graph(data,data.loc[0,start_time],data.loc[0,start_time]+timedelta(minutes=15))
    # print(graph['E 6 St & Avenue D'])
    # overflow = {node:{} for node in graph.nodes() if graph.nodes[node]['type']=='overflow'}
    # underflow = {node:{} for node in graph.nodes() if graph.nodes[node]['type']=='underflow'}
    # print('edgesss',graph.edges)
    # print('vertixxx',graph.nodes)
    #

    # cgraph = cloned_station_vertices(graph)
    data = pd.read_csv('data/202105-capitalbikeshare-tripdata.csv')
    data = data.loc[[type(i) == str for i in data[start_station_name]]]
    data = data.loc[[type(i) == str for i in data[end_station_name]]]
    # data = data.loc[data.index[:10000]]
    data[start_time] = data[start_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    data[end_time] = data[end_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    start = datetime(day=1, month=5, year=2021)
    for i in range(60):
        if start.weekday() in [0, 6]:
            run_test(data, name='capital', interval=15, begin_interval=start, end_interval=start + timedelta(hours=10),
                     trial='weekend' + str(i))
        else:
            run_test(data, name='capital', interval=15, begin_interval=start, end_interval=start + timedelta(hours=10),
                     trial='weekday' + str(i))
        if i%2 == 1:
            start += timedelta(days=1)


    data = pd.read_csv('data/202105-citibike-tripdata.csv')
    data = data.loc[[type(i) == str for i in data[start_station_name]]]
    data = data.loc[[type(i) == str for i in data[end_station_name]]]
    # data = data.loc[data.index[:10000]]
    data[start_time] = data[start_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    data[end_time] = data[end_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    start = datetime(day=1, month=5, year=2021)
    for i in range(60):
        if start.weekday() in [0, 6]:
            run_test(data, name='citi', interval=1, begin_interval=start, end_interval=start + timedelta(hours=10),
                     trial='weekend' + str(i))
        else:
            run_test(data, name='citi', interval=1, begin_interval=start, end_interval=start + timedelta(hours=10),
                     trial='weekday' + str(i))
        if i%2 == 1:
            start += timedelta(days=1)


    # ur = UGA_RSL()
    # print(ur.find_optimal(cwgraph))
    # u = UGA()
    # print(u.find_optimal(cwgraph))
