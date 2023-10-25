from workload import workload_allkway, root_mean_squared_error
import numpy as np
import time


def time_RMSE(n_list, d_list, repeat=5):
    for n in n_list:
        for d in d_list:
            print("-----------------------------------------")
            print(n, d)
            time_ls = []
            loss_ls = []
            for r in range(repeat):
                start = time.time()
                system, num_query = workload_allkway(n, d, 3, choice="sumvar")
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


def time_MaxVar(n_list, d_list, repeat=5, solver="cvxpy"):
    for n in n_list:
        for d in d_list:
            print("-----------------------------------------")
            print(n, d)
            time_ls = []
            loss_ls = []
            for r in range(repeat):
                start = time.time()
                system, num_query = workload_allkway(n, d, 3, choice="maxvar")
                max_var = system.get_noise_level(solver=solver, time_limit=10)
                end = time.time()

                time_ls.append(end-start)
                loss_ls.append(max_var)

            mean_time = np.mean(time_ls)
            mean_loss = np.mean(loss_ls)
            std_time = np.std(time_ls)
            std_loss = np.std(loss_ls)
            print("Time: ", mean_time, 2*std_time)
            print("Loss: ", mean_loss, 2*std_loss)


if __name__ == '__main__':
    n_list = [2**i for i in range(1, 11)]
    d_list = [5]
    # time_RMSE(n_list, d_list, repeat=5)
    time_MaxVar(n_list, d_list, repeat=1, solver="cvxpy")

