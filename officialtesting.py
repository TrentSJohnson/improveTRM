from datetime import timedelta, datetime

from graphprocessing import *
from models.hs import RSL, TRM_RSL, RSLL
from models.local_ratio import Local_Ratio
from models.trm import TRM
from models.uga import UGA_RSL, UGA, RSL1


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
        return (temp, tempt, len(list(graph.nodes)), start_time_dt, get_shortest_assignment(graph))

    def test(self, algorithms, data, interval, start_time_='started_at', epsilon=1.0, radius=500, trials=10):
        self.scores = []
        # epochs = int((max(data.starttime)-start_time).total_seconds()/(60*15))
        self.runtimes = []
        self.graphs = []
        self.times = []
        results = []
        for i in range(trials):
            results.append(self.test_(algorithms, data, interval, start_time_, epsilon, radius))
        scores = [results[i][0] for i in range(len(results))]
        runtimes = [results[i][1] for i in range(len(results))]
        graph_sizes = [results[i][2] for i in range(len(results))]
        times = [results[i][3] for i in range(len(results))]
        sdists = [results[i][4] for i in range(len(results))]
        return scores, runtimes, times, graph_sizes, sdists


def run_test(datam, name='citi'):
    testing = Testing()
    scores, runtimes, times, graphs, sdists = testing.test(
        [UGA(), UGA_RSL(), RSL1(), Local_Ratio(), TRM(), RSL(), TRM_RSL(), RSLL()],
        data, interval=1, epsilon=0.2, trials=5)

    scores_df = pd.DataFrame(np.abs(scores),
                             columns=['UGA', 'UGA_RSL', 'RSL1', 'Local_Ratio', 'TRM', 'RSL', 'TRM_RSL', 'RSLL'])
    runtimes_df = pd.DataFrame(np.abs(runtimes),
                               columns=['UGA', 'UGA_RSL', 'RSL1', 'Local_Ratio', 'TRM', 'RSL', 'TRM_RSL', 'RSLL'])
    scores_df.to_csv('outputs/scores_' + name + '.csv')
    runtimes_df.to_csv('outputs/runtimes_' + name + '.csv')
    times_df = pd.DataFrame(times)
    graphs_df = pd.DataFrame(graphs)
    times_df.to_csv('outputs/times_' + name + '.csv')
    graphs_df.to_csv('outputs/graphs_' + name + '.csv')
    sdists_df = pd.DataFrame(sdists)
    sdists_df.to_csv('outputs/sdists_citi.csv')


def metrodate(s):
    date, time = s.split(' ')
    month, day, year = date.split('/')

    hour, minute = time.split(':')
    return {'month': int(month), 'day': int(day), 'year': int(year), 'hour': int(hour), 'minute': int(minute)}

if __name__ == '__main__':
    data = pd.read_csv('data/202105-citibike-tripdata.csv')
    data = data.loc[[type(i) == str for i in data[start_station_name]]]
    data = data.loc[[type(i) == str for i in data[end_station_name]]]
    # data = data.loc[data.index[:10000]]
    data[start_time] = data[start_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    data[end_time] = data[end_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    metro = pd.read_csv('data/metro-trips-2021-q1.csv')
    metro[start_time] = metro['start_time'].apply(lambda x: datetime(**metrodate(x)))
    metro[end_time] = metro['end_time'].apply(lambda x: datetime(**metrodate(x)))
    metro[start_station_name] = metro['start_station']
    metro[end_station_name] = metro['end_station']
    metro[start_lat] = metro['start_lat']
    metro[start_lng] = metro['start_lon']
    metro[end_lng] = metro['end_lon']
    metro[end_lng] = metro['end_lon']
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

    run_test(metro, name='metro')
    run_test(data, name='citi')
    run_test(metro)

    # ur = UGA_RSL()
    # print(ur.find_optimal(cwgraph))
    # u = UGA()
    # print(u.find_optimal(cwgraph))
