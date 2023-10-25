from class_resplan import ResidualPlanner
import numpy as np
import itertools
import time
import pandas as pd


def test_allkway_csv(n, d=5, k=3):
    domains = [n] * d
    col_names = [str(i) for i in range(d)]
    system = ResidualPlanner(domains)
    data_numpy = np.zeros([10_000, d])
    df = pd.DataFrame(data_numpy, columns=col_names)
    system.input_data(df, col_names)

    att = tuple(range(len(domains)))
    total = 0
    for i in range(0, k+1):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for t, subset in enumerate(subset_i):
            system.input_mech(subset, 1)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
            if t % 10_000 == 0 and t > 0:
                print("Selecting marginal: ", t)
    # print("total num of queries: ", total, "\n")
    return system, total


def time_reconstruction(n_list, d_list, repeat=5):
    for n in n_list:
        for d in d_list:
            print("-------------------------------------------------------------")
            print("n = ", n, " d = ", d)
            reconstruct_time = []

            start = time.time()
            system, num_query = test_allkway_csv(n, d, 3)
            sum_var = system.selection(choice="sumvar")
            time_select = time.time()
            print("time_select: ", time_select - start, "\n")

            system.measurement()
            time_measure = time.time()
            print("time_measure:", time_measure - time_select, "\n")
            for k in range(repeat):
                start = time.time()
                system.reconstruction()
                end = time.time()
                reconstruct_time.append(end - start)
                print("time_reconstruction: ", end - start, "\n")

            mean = np.mean(reconstruct_time)
            std = np.std(reconstruct_time)
            print("Time mean value and 2*std: ", mean, 2*std)


if __name__ == '__main__':
    n_list = [2**i for i in range(1, 11)]
    d_list = [5]
    time_reconstruction(n_list, d_list, 5)
