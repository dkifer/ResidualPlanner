from workload import workload_large_dataset, root_mean_squared_error
import numpy as np
import time


def loss_RMSE(dataset_list, workload_list, repeat=1):
    for dataset in dataset_list:
        for workload in workload_list:
            print("-----------------------------------------")
            print(dataset, workload)
            time_ls = []
            loss_ls = []
            for r in range(repeat):
                start = time.time()
                system, num_query = workload_large_dataset(dataset, workload, choice="sumvar")
                sum_var = system.get_noise_level()
                rmse = root_mean_squared_error(sum_var, num_query, pcost=1)
                end = time.time()

                time_ls.append(end-start)
                loss_ls.append(rmse)

            mean_time = np.mean(time_ls)
            std_time = np.std(time_ls)
            mean_loss = np.mean(loss_ls)
            std_loss = np.std(loss_ls)
            print("Time: ", mean_time, 2*std_time)
            print("Loss: ", mean_loss, 2*std_loss)


def loss_MaxVar(dataset_list, workload_list, repeat=1, solver="cvxpy", time_limit=10):
    for dataset in dataset_list:
        for workload in workload_list:
            print("-----------------------------------------")
            print(dataset, workload)
            time_ls = []
            loss_ls = []
            if dataset == "Loans" and workload == 5:
                time_limit = 30

            for r in range(repeat):
                start = time.time()
                system, num_query = workload_large_dataset(dataset, workload, choice="maxvar")
                max_var = system.get_noise_level(solver=solver, time_limit=time_limit)
                end = time.time()

                time_ls.append(end-start)
                loss_ls.append(max_var)

            mean_time = np.mean(time_ls)
            std_time = np.std(time_ls)
            mean_loss = np.mean(loss_ls)
            std_loss = np.std(loss_ls)
            print("Time: ", mean_time, 2*std_time)
            print("Loss: ", mean_loss, 2*std_loss)


if __name__ == '__main__':
    dataset_list = ["CPS", "Adult", "Loans"]
    workload_list = [1, 2, 3, 4, 5, "3D", "Small"]
    # loss_RMSE(dataset_list, workload_list, repeat=1)
    loss_MaxVar(dataset_list, workload_list, repeat=1, solver="cvxpy")
    # loss_MaxVar(dataset_list, workload_list, repeat=1, solver="gurobi", time_limit=10)

