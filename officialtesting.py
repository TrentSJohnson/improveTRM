from datetime import timedelta, datetime

from tqdm import tqdm

from graphprocessing import *
from models.hs import TRM_RLS, RLS
from models.local_ratio import Local_Ratio
from models.trm import TRM
from models.uga import UGA_RLS


class Testing:
    def test_(self, algorithms, data, interval, epsilon, radius, begin_interval, end_interval):
        print('started thread')
        filtered_data = data

        if not begin_interval is None:
            filtered_data = filtered_data[(filtered_data[start_time] >= begin_interval)].dropna()
        if not end_interval is None:
            filtered_data = filtered_data[(filtered_data[start_time] >= begin_interval)].dropna()
        if len(filtered_data[start_time]) < 10:
            return None
        start_time_dt = filtered_data[start_time][np.random.choice(filtered_data.index)]

        td = timedelta(minutes=interval)
        graph = cloned_station_vertices(build_station_graph(data, start_time_dt, start_time_dt + td))
        n_worker = filtered_data[(filtered_data[start_time] >= start_time_dt) &
                                 (filtered_data[start_time] <= start_time_dt + td)].shape[0]
        m=1
        while len(graph.nodes) < 75 or len(graph.nodes)+n_worker > 200:
            print('failed',len(graph.nodes))
            start_time_dt = data[start_time][np.random.choice(data.index)]
            m *= 5/ 4 if len(graph.nodes) < 75 else   3/4
            graph = cloned_station_vertices(build_station_graph(data, start_time_dt, start_time_dt + td))
            n_worker = filtered_data[(filtered_data[start_time] >= start_time_dt) &
                                     (filtered_data[start_time] <= start_time_dt + td)].shape[0]
            td = timedelta(minutes=m*interval)


        results = []
        for e in [1,.8,.6,.4,.2]:
            cwgraph = build_cwgraph(data, start_time_dt, start_time_dt + td, radius=radius, epsilon=e)
            print('nodes:', len(cwgraph.nodes))
            temp = []
            tempt = []
            short = get_shortest_assignment(cwgraph)
            print('shortest:', short)
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
            results.append((temp, tempt, len(list(cwgraph.nodes)), start_time_dt, short))
        return results

    def test(self, algorithms, data, interval, epsilon=1.0, radius=500, begin_interval=None,
             end_interval=None):
        self.scores = []
        self.runtimes = []
        self.graphs = []
        self.times = []
        results = self.test_(algorithms, data, interval, epsilon, radius, begin_interval, end_interval)

        scores = [results[r_][0] for r_ in range(len(results))]
        runtimes = [results[r_][1] for r_ in range(len(results))]
        graph_sizes = [results[r_][2] for r_ in range(len(results))]
        times = [results[r_][3] for r_ in range(len(results))]
        sdists = [results[r_][4] for r_ in range(len(results))]

        return scores, runtimes, times, graph_sizes, sdists


def run_test(data, name='citi', interval=1, begin_interval=None, end_interval=None, trial='', epsilon=.2):
    testing = Testing()
    scores, runtimes, times, graphs, sdists = testing.test([TRM(), Local_Ratio(), RLS(), UGA_RLS(), TRM_RLS()],
                                                           data, interval=interval, epsilon=epsilon,
                                                           begin_interval=begin_interval, end_interval=end_interval)

    cols = ['TRM', 'Local_Ratio', 'RLS', 'UGA_RLS', 'TRHS']
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
    data = pd.read_csv('data/202105-citibike-tripdata.csv')
    data = data.loc[[type(i) == str for i in data[start_station_name]]]
    data = data.loc[[type(i) == str for i in data[end_station_name]]]
    # data = data.loc[data.index[:10000]]
    data[start_time] = data[start_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    data[end_time] = data[end_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    start = datetime(day=1, month=5, year=2021, hour=7)
    for i in tqdm(range(15)):
        if start.weekday() in [0, 6]:
            run_test(data, name='citi', interval=1, begin_interval=start, end_interval=start + timedelta(hours=12),
                     trial='weekend' + str(i))
        else:
            run_test(data, name='citi', interval=1, begin_interval=start, end_interval=start + timedelta(hours=12),
                     trial='weekday' + str(i))
        if i % 2 < 2:
            start += timedelta(days=1)
            """
    data = pd.read_csv('data/202105-bluebikes-tripdata.csv')
    data[start_station_name] = data['start station name']
    data[end_station_name] = data['end station name']
    data[start_lat] = data['start station latitude']
    data[start_lng] = data['start station longitude']
    data[end_lat] = data['end station latitude']
    data[end_lng] = data['end station longitude']
    data[start_time] = data['starttime']
    data[end_time] = data['stoptime']
    data = data.loc[[type(i) == str for i in data[start_station_name]]]
    data = data.loc[[type(i) == str for i in data[end_station_name]]]
    # data = data.loc[data.index[:10000]]
    data[start_time] = data[start_time].apply(lambda x: datetime.strptime(x.split('.')[0], "%Y-%m-%d %H:%M:%S"))
    data[end_time] = data[end_time].apply(lambda x: datetime.strptime(x.split('.')[0], "%Y-%m-%d %H:%M:%S"))
    start = datetime(day=1, month=5, year=2021, hour=7)

    for i in tqdm(range(8,15)):
        if start.weekday() in [0, 6]:
            run_test(data, name='blue', interval=10, begin_interval=start, end_interval=start + timedelta(hours=12),
                     trial='weekend' + str(i))
        else:
            run_test(data, name='blue', interval=10, begin_interval=start, end_interval=start + timedelta(hours=12),
                     trial='weekday' + str(i))
        if i % 2 < 2:
            start += timedelta(days=1)

    data = pd.read_csv('data/202105-capitalbikeshare-tripdata.csv')
    data = data.loc[[type(i) == str for i in data[start_station_name]]]
    data = data.loc[[type(i) == str for i in data[end_station_name]]]
    # data = data.loc[data.index[:10000]]
    data[start_time] = data[start_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    data[end_time] = data[end_time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    start = datetime(day=1, month=5, year=2021, hour=7)
    for i in tqdm(range(15)):
        if start.weekday() in [0, 6]:
            run_test(data, name='capital', interval=10, begin_interval=start, end_interval=start + timedelta(hours=12),
                     trial='weekend' + str(i))
        else:
            run_test(data, name='capital', interval=10, begin_interval=start, end_interval=start + timedelta(hours=12),
                     trial='weekday' + str(i))
        if i % 2 < 2:
            start += timedelta(days=1)

