import numpy as np
import itertools
import time
from cdp2adp import *
import pandas as pd
from class_resplan import ResidualPlanner


def test_Adult_small():
    domains = [16, 32, 9, 32, 16, 7, 15, 6, 5, 2, 32, 32, 32, 42, 2]
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country', 'income>50K']
    system = ResidualPlanner(domains)
    data = pd.read_csv("adult_small.csv")
    system.input_data(data, col_names)
    # print("Len of adult dataset: ", len(data))

    att = tuple(range(len(domains)))
    total = 0
    for i in range(3, 4):
        subset_i = list(itertools.combinations(att, i))
        # print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset, var_bound=1)
            cur_domains = [domains[c] for c in subset]
            total += np.prod(cur_domains)
    # print("Total num of queries: ", total, "\n")
    return system, total


def compare_rho_empirical_error():
    start = time.time()
    ep_ls = [0.03, 0.1, 0.31, 1.0, 3.16, 10] 

    for eps in ep_ls:
        print("------------------- ep: ", eps, "------------------")
        delta = 1e-9
        rho = cdp_rho(eps, delta)
        pcost = rho * 2

        system, total = test_Adult_small()
        sum_var = system.selection(choice="sumvar", pcost=pcost)
        system.measurement()
        system.reconstruction()
        ord = 2
        l_error = system.get_mean_error(ord=ord)
        nonneg_error = system.get_error_nonneg(ord=ord, consist=False)
        consist_error = system.get_error_nonneg(ord=ord, consist=True)
        print("Mean Error: ", l_error)
        print("Nonneg Error: ", nonneg_error)
        print("Consistent Error: ", consist_error)

    end = time.time()
    print("time needed: ", end-start)


def test_two_one_way():
    domains = [2, 3]
    d = 2
    col_names = [str(i) for i in range(d)]
    system = ResidualPlanner(domains)
    data_numpy = np.zeros([10_000, d])
    df = pd.DataFrame(data_numpy, columns=col_names)
    system.input_data(df, col_names)
    system.input_mech((0,))
    system.input_mech((1,))

    pcost = 2
    sum_var = system.selection(choice="sumvar", pcost=pcost)
    system.measurement()
    system.reconstruction()
    print("sum-of-var is:", sum_var)

    return


if __name__ == '__main__':
    # compare_rho_empirical_error()
    R = np.array([[1, -1, 0, 0],
                  [1, 0, -1, 0],
                  [1, 0, 0, -1]])
    S = R @ R.T
    S_inv = np.linalg.inv(S)
    e1 = R[:, 1]
    e2 = R[:, 2]
    p_mat = R.T @ S_inv @ R
    print("diagonal under unbounded dp", np.diag(p_mat))

    diff = np.reshape(e1-e2, [3, 1])
    diff_cost = diff.T @ S_inv @ diff

    diff_R = np.array([[2, 1, 1, -1, -1, 0],
                       [1, 2, 1, 1, 0, -1],
                       [1, 1, 2, 0, 1, 1]])
    diff_p_mat = diff_R.T @ S_inv @ diff_R
    print("diagonal under bounded dp", np.diag(diff_p_mat))
