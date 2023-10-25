import numpy as np
import itertools
from class_sumvar import RPSumVar
from class_maxvar import RPMaxVar


def root_mean_squared_error(sum_var, num_query, pcost):
    rmse = np.sqrt(sum_var / num_query) / pcost
    return rmse


def workload_allkway(n, d, k, choice="sumvar"):
    domains = [n for _ in range(d)]
    if choice == "sumvar":
        system = RPSumVar(domains)
    elif choice == "maxvar":
        system = RPMaxVar(domains)
    else:
        print("Invalid choice, choose between sumvar and maxvar.")
        return

    att = tuple(range(len(domains)))
    total = 0
    for i in range(0, k+1):
        subset_i = list(itertools.combinations(att, i))
        # print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c] + 0.0 for c in subset]
            total += np.multiply.reduce(cur_domains)
    return system, total


def dataset_domains(dataset):
    """Return dataset domain size for each attribute."""
    if dataset == "CPS":
        return [50, 100, 7, 4, 2]
    elif dataset == "Adult":
        return [85, 9, 100, 16, 7, 15, 6, 5, 2, 100, 100, 99, 42, 2]
    elif dataset == "Loans":
        return [101, 101, 101, 101, 3, 8, 36, 6, 51, 4, 5, 15]
    else:
        print("Invalid Choice! Please choose between CPS, Adult and Loans")
        return []


def workload_large_dataset(dataset, workload, choice="sumvar"):
    """Return system given workload.

    dataset: Choose between "CPS", "Adult", "Loans"
    workload: k       --> k way marginals
              "3D"    --> All 0, 1, 2, 3 way marginals
              "Small" --> All marginals with size <= 5000
    """
    domains = dataset_domains(dataset)
    if choice == "sumvar":
        system = RPSumVar(domains)
    elif choice == "maxvar":
        system = RPMaxVar(domains)
    else:
        print("Invalid choice, choose between sumvar and maxvar.")
        return
    num_att = len(domains)
    att = tuple(range(num_att))

    if type(workload) == int:
        lower = workload
        upper = lower + 1
    elif workload == "3D":
        lower = 0
        upper = 4
    elif workload == "Small":
        lower = 0
        upper = num_att + 1
    else:
        print("Invalid workload, choose between All, 3D, Small")
        return

    total = 0
    for i in range(lower, upper):
        subset_i = list(itertools.combinations(att, i))
        # print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            cur_domains = [domains[c] + 0.0 for c in subset]
            num_query = np.multiply.reduce(cur_domains)
            if workload == "Small":
                if 0 < num_query <= 5000:
                    system.input_mech(subset, 1)
                    total += num_query
            if workload == "3D" or type(workload) == int:
                system.input_mech(subset, 1)
                total += num_query
    return system, total
