from datetime import timedelta, datetime

from tqdm import tqdm
import cProfile
from graphprocessing import *
from models.hs import TRM_RLS, RLS
from models.local_ratio import Local_Ratio
from models.trm import TRM
from models.uga import UGA_RLS


class Testing:
    def test_(self, algorithms, data, interval, ratios, radius, begin_interval, end_interval, graph_min=100,
              graph_max=400):
        print('started thread')
        filtered_data = data

        if not begin_interval is None:
            filtered_data = filtered_data[(
                filtered_data[start_time] >= begin_interval)].dropna()
        if not end_interval is None:
            filtered_data = filtered_data[(
                filtered_data[start_time] >= begin_interval)].dropna()
        start_time_dt = filtered_data[start_time][np.random.choice(
            filtered_data.index)]

        td = timedelta(minutes=interval)
        cwgraph_min = build_cwgraph(
            data, start_time_dt, start_time_dt + td, radius=radius, ratio=min(ratios))
        cwgraph_max = build_cwgraph(
            data, start_time_dt, start_time_dt + td, radius=radius, ratio=max(ratios))
        m = 1
        tries = 0
        while len(cwgraph_min.nodes) < graph_min or len(cwgraph_max.nodes) > graph_max:
            tries += 1
            if tries == 10:
                tries = 0
                start_time_dt = filtered_data[start_time][np.random.choice(
                    filtered_data.index)]
            print('failed min:', len(cwgraph_min.nodes))
            print('failed max:', len(cwgraph_max.nodes))
            m *= graph_min/len(cwgraph_min.nodes) if len(cwgraph_min.nodes) < graph_min else graph_max/len(cwgraph_max.nodes)
            td = timedelta(minutes=m * interval)
            cwgraph_min = build_cwgraph(
                data, start_time_dt, start_time_dt + td, radius=radius, ratio=min(ratios))
            cwgraph_max = build_cwgraph(
                data, start_time_dt, start_time_dt + td, radius=radius, ratio=max(ratios))

        results = []
        for r in ratios:
            cwgraph = build_cwgraph(
                data, start_time_dt, start_time_dt + td, radius=radius, ratio=r)
            print('nodes:', len(cwgraph.nodes))
            temp = []
            tempt = []
            short = get_shortest_assignment(cwgraph)
            print('shortest:', short)
            for algo in algorithms:
                if len(list(cwgraph.nodes)) < 500:
                    print(str(algo))
                    start = datetime.now()
                    graph = algo.test(cwgraph)[0]
                    if not check_fully_assigned(graph, cwgraph):
                        raise Exception('Not fully assigned')
                    graph = complete_graph(graph, cwgraph)
                    score = score_graph(graph, cwgraph=cwgraph)
                    temp.append(score)
                    tempt.append((datetime.now() - start).total_seconds())
                    print(score)
                else:
                    temp.append(np.nan)
                    tempt.append(np.nan)
            results.append(
                (temp, tempt, len(list(cwgraph.nodes)), start_time_dt, short))
        return results

    def test(self, algorithms, data, interval, ratios=None, radius=500, begin_interval=None,
             end_interval=None):
        if ratios is None:
            ratios = [1.0]
        self.scores = []
        self.runtimes = []
        self.graphs = []
        self.times = []
        results = self.test_(algorithms, data, interval,
                             ratios, radius, begin_interval, end_interval)

        scores = [results[r_][0] for r_ in range(len(results))]
        runtimes = [results[r_][1] for r_ in range(len(results))]
        graph_sizes = [results[r_][2] for r_ in range(len(results))]
        times = [results[r_][3] for r_ in range(len(results))]
        sdists = [results[r_][4] for r_ in range(len(results))]

        return scores, runtimes, times, graph_sizes, sdists


def run_test(data, name='citi', interval=1, begin_interval=None, end_interval=None, trial='', ratios=None):
    if ratios is None:
        ratios = [1.0]
    testing = Testing()
    scores, runtimes, times, graphs, sdists = testing.test([TRM(), RLS(), TRM_RLS(), UGA_RLS(), Local_Ratio()],
                                                           data, interval=interval, ratios=ratios,
                                                           begin_interval=begin_interval, end_interval=end_interval)

    cols = ['TRM', 'RLS', 'TRHS', 'GHS', 'LR']
    scores_df = pd.DataFrame(np.abs(scores), columns=cols, index=ratios)
    runtimes_df = pd.DataFrame(np.abs(runtimes), columns=cols)
    scores_df.to_csv('outputs/scores_' + trial + name + '.csv')
    runtimes_df.to_csv('outputs/runtimes_' + trial + name + '.csv')
    times_df = pd.DataFrame(times, index=ratios)
    graphs_df = pd.DataFrame(graphs, index=ratios)
    times_df.to_csv('outputs/times_' + trial + name + '.csv')
    graphs_df.to_csv('outputs/graphs_' + trial +  name + '.csv')
    sdists_df = pd.DataFrame(sdists)
    sdists_df.to_csv('outputs/sdists_' + trial + name + '.csv')


def metrodate(s):
    date, time = s.split(' ')
    month, day, year = date.split('/')

    hour, minute = time.split(':')
    return {'month': int(month), 'day': int(day), 'year': int(year), 'hour': int(hour), 'minute': int(minute)}


def default_run_test(data, name, start_trial_id, start):
    """
    Run a test on a data set with default values

    Args:
        data: pd.DataFrame
            data set to be tested
        name: str
            name of the data set
        start_trial_id: int
            id of the first trial
        start: datetime.datetime
            start date of the data set
    """
    ratios = [1/5,1/4, 1/3, 1/2, 1, 2, 3, 4,5]
    start_ = datetime(day=start.day, month=start.month,
                      year=start.year, hour=start.hour)

    for i in range(start_trial_id, 1 + start_trial_id):
        if start_.weekday() in [0, 6]:
            run_test(data, name=name, interval=10, begin_interval=start_, end_interval=start_ + timedelta(hours=12),
                     trial='weekend' + str(i), ratios=ratios)
        else:
            run_test(data, name=name, interval=10, begin_interval=start_, end_interval=start_ + timedelta(hours=12),
                     trial='weekday' + str(i), ratios=ratios)
        if i % 2 == 2:
            start_ += timedelta(days=1)


def main():
    # process citibike data
    citi = pd.read_csv('data/202105-citibike-tripdata.csv',
                       parse_dates=[start_time, end_time])
    print(type(citi.loc[0, start_time]), citi.loc[0, start_time])
    citi = citi.loc[[type(i) == str for i in citi[start_station_name]]]
    citi = citi.loc[[type(i) == str for i in citi[end_station_name]]]

    # process blue data
    blue = pd.read_csv('data/202105-bluebikes-tripdata.csv',
                       parse_dates=['starttime', 'stoptime'])
    blue[start_station_name] = blue['start station name']
    blue[end_station_name] = blue['end station name']
    blue[start_lat] = blue['start station latitude']
    blue[start_lng] = blue['start station longitude']
    blue[end_lat] = blue['end station latitude']
    blue[end_lng] = blue['end station longitude']
    blue[start_time] = blue['starttime']
    blue[end_time] = blue['stoptime']
    blue = blue.loc[[type(i) == str for i in blue[start_station_name]]]
    blue = blue.loc[[type(i) == str for i in blue[end_station_name]]]

    # process capital data
    capit = pd.read_csv('data/202105-capitalbikeshare-tripdata.csv',
                        parse_dates=[start_time, end_time])
    capit = capit.loc[[type(i) == str for i in capit[start_station_name]]]
    capit = capit.loc[[type(i) == str for i in capit[end_station_name]]]

    # run tests
    start_id = 0
    num_trials = 30
    for trial_id in tqdm(range(start_id, start_id + num_trials)):
        start = datetime(day=np.random.randint(
            1, 30), month=5, year=2021, hour=7)
        for data, name in [(citi, 'citi'), (blue, 'blue'), (capit, 'capit')]:
            default_run_test(data, name, trial_id, start)

if __name__ == '__main__':
    main()