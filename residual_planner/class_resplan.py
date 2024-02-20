import numpy as np
import itertools
import scipy.sparse as sp
import pandas as pd
import cvxpy as cp
import time
from scipy.sparse import csr_matrix
from cdp2adp import *


"""
domains = [3, 3, 3 ,3 ,3]
5 attributes, each attribute 3 values
-------------------------------------------
att = ()        --> sum query
att = (0)       --> marginal A
att = (2, 3)    --> marginal CD
att = (0, 1 ,2) --> marginal ABC
"""


def mult_kron_vec(mat_ls, vec):
    """Fast Kronecker matrix vector multiplication."""
    V = vec.reshape(-1, 1)
    row = 1
    X = V.T
    for Q in mat_ls[::-1]:
        m, n = Q.shape
        row *= m
        X = Q.dot(X.reshape(-1, n).T)
    return X.reshape(row, -1)


def find_var_max(coeff, A, b, pcost):
    """Solve the fitness-for-use optimization problem.

    min sum(coeff / x)
    s.t. A x <= b
    """
    size = len(coeff)
    x = cp.Variable(size)
    constraints = [x >= 0]
    constraints += [A @ x - b <= 0]
    obj = cp.Minimize(cp.sum(coeff @ cp.inv_pos(x)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return x.value / pcost, obj.value * pcost


def find_var_sum_cauchy(var, pcoeff, c):
    """Solve the sum-of-variance optimization problem.

    min sum(var @ x)
    s.t. sum(pcoeff @ 1/x) == c
    """
    T = np.sum(np.sqrt(var * pcoeff))**2 / c
    x = np.sqrt(T * pcoeff / (c * var))
    return x, T


def subtract_matrix(k, is_sparse=True):
    """Return Subtraction matrix Sub_k."""
    mat = np.zeros([k-1, k])
    for i in range(k-1):
        mat[i, i] = 1
        mat[i, i+1] = -1
    if is_sparse:
        return sp.csr_matrix(mat)
    else:
        return mat


def all_subsets(att):
    """Return all subsets of a tuple."""
    length = len(att)
    subsets = [()]
    for i in range(1, length + 1):
        subset_i = list(itertools.combinations(att, i))
        subsets = subsets + subset_i
    return subsets


class Mechanism:

    def __init__(self, domains, att, var_bound):
        self.domains = domains
        self.num_att = len(domains)
        self.att = att
        self.var_bound = var_bound
        self.covar = None
        self.noisy_answer = None
        cur_domains = [self.domains[at] for at in att]
        self.num_queries = np.prod([c + 0.0 for c in cur_domains])
        self.variance = None
        self.true_answer = None
        pass

    def output_bound(self):
        return self.var_bound

    def input_noisy_answer(self, answer):
        self.noisy_answer = answer

    def input_true_answer(self, answer):
        self.true_answer = answer

    def get_num_queries(self):
        return self.num_queries

    def input_variance(self, var):
        self.variance = var

    def output_variance(self):
        return self.variance

    def get_noisy_answer(self):
        return self.noisy_answer

    def get_true_answer(self):
        return self.true_answer

    pass


class ResMech:

    def __init__(self, domains, att):
        self.domains = domains
        self.num_att = len(self.domains)
        self.att = att
        self.core_mat = None
        self.res_mat_list = []
        self.get_core_matrix()

        self.noise_level = None
        self.covar = None
        self.calculated = False
        self.recon_answer = None
        self.noisy_answer = None
        self.true_answer = None
        self.true_recon_answer = None

    def get_core_matrix(self):
        att_set = set(list(self.att))
        for i in range(0, self.num_att):
            att_size = self.domains[i]
            if i in att_set:
                res_mat = subtract_matrix(att_size)
                self.res_mat_list.append(res_mat)

    def input_noise_level(self, noise_level):
        self.noise_level = noise_level
        self.calculated = True

    def output_noise_level(self):
        if self.calculated:
            return self.noise_level
        else:
            print("Not yet calculated!")
            return 0.0

    def is_calculated(self):
        return self.calculated

    def input_data(self, data, col_names):
        self.data = data
        self.col_names = col_names
        pass

    def measure(self):
        sub_domains = [self.domains[at]+0.0 for at in self.att]
        bins = [np.arange(t+1) for t in sub_domains]
        if self.att == ():
            sparse_vec = np.array(len(self.data))
        else:
            datavector = np.histogramdd(self.data.values, bins)[0]
            datavector = datavector.flatten()
            sparse_vec = csr_matrix(datavector)
        true_answer = mult_kron_vec(self.res_mat_list, sparse_vec)
        col_size = np.prod(sub_domains).astype(int)
        rd = np.sqrt(self.noise_level) * np.random.normal(size=[col_size, 1])
        cov_rd = mult_kron_vec(self.res_mat_list, rd)
        self.noisy_answer = true_answer + cov_rd
        # todo: move it to class Mechnism
        self.true_answer = true_answer + np.zeros_like(cov_rd)

    def get_recon_answer(self, mat_list):
        self.recon_answer = mult_kron_vec(mat_list, self.noisy_answer)
        return self.recon_answer

    def get_origin_answer(self, mat_list):
        self.true_recon_answer = mult_kron_vec(mat_list, self.true_answer)
        return self.true_recon_answer


class ResidualPlanner:

    def __init__(self, domains):
        self.domains = domains
        self.col_names = None
        self.data = None
        self.num_of_mech = 0
        self.num_of_res = 0

        self.mech_index = {}
        self.res_index = {}
        self.id2res = {}

        self.mech_dict = {}
        self.res_dict = {}
        self.pcost_coeff = {}
        self.var_bound = {}

        self.var_coeff = {}
        self.sigma_square = None
        pass

    def input_mech(self, att, var_bound=1.0):
        mech = Mechanism(self.domains, att, var_bound)
        self.mech_dict[att] = mech
        self.mech_index[att] = self.num_of_mech
        self.num_of_mech += 1
        self.var_bound[att] = var_bound

        att_subsets = all_subsets(att)
        for subset in att_subsets:
            if subset not in self.res_dict:
                sub_domains = [self.domains[at] for at in subset]
                pcost_coeff = np.prod([(c - 1) / c for c in sub_domains])
                self.pcost_coeff[subset] = pcost_coeff
                res_mech = ResMech(self.domains, subset)
                self.res_dict[subset] = res_mech
        for subset in att_subsets:
            sub_domains = [self.domains[at] for at in subset]
            # be careful of the numerical overflow
            var_coeff = np.prod([(c - 1) / c for c in sub_domains])
            div_list = []
            for at in att:
                if at not in subset:
                    div_list.append(1.0 / self.domains[at] ** 2)
            divisor = np.prod(div_list)
            var_coeff = var_coeff * divisor

            self.var_coeff[att, subset] = var_coeff

    def input_data(self, data, col_names):
        self.data = data
        self.col_names = col_names
        pass

    def get_coeff_maxvar(self):
        pcost_coeff_list = []
        var_coeff_list = []
        row_list = []
        col_list = []
        var_bound_list = []
        res2id = {}
        res_id = 0
        mech_id = 0
        for res_att in self.res_dict.keys():
            pcost_coeff_list.append(self.pcost_coeff[res_att])
            res2id[res_att] = res_id
            self.id2res[res_id] = res_att
            res_id += 1
        for mech_att in self.mech_dict.keys():
            subsets = all_subsets(mech_att)
            for res_att in subsets:
                var_coeff_list.append(self.var_coeff[mech_att, res_att])
                row_list.append(mech_id)
                col_list.append(res2id[res_att])
            mech_id += 1
            var_bound_list.append(self.var_bound[mech_att])
        A = sp.csr_matrix((var_coeff_list, (row_list, col_list)), shape=(
            len(self.mech_dict), len(self.res_dict)))
        b = np.array(var_bound_list)
        coeff = np.array(pcost_coeff_list)
        return coeff, A, b

    def get_coeff_sum_var(self):
        pcost_coeff_list = []

        res2id = {}
        res_id = 0
        for res_att in self.res_dict.keys():
            pcost_coeff_list.append(self.pcost_coeff[res_att])
            res2id[res_att] = res_id
            self.id2res[res_id] = res_att
            res_id += 1

        pcost_coeff = np.array(pcost_coeff_list)
        var_coeff = np.zeros_like(pcost_coeff)
        for mech_att in self.mech_dict.keys():
            subsets = all_subsets(mech_att)
            mech = self.mech_dict[mech_att]
            for res_att in subsets:
                idx = res2id[res_att]
                num_of_queries = mech.get_num_queries()
                var_coeff[idx] += self.var_coeff[mech_att, res_att] * num_of_queries
        return var_coeff, pcost_coeff

    def selection(self, choice='sumvar', pcost=1):
        if choice == 'sumvar':
            var_coeff, pcost_coeff = self.get_coeff_sum_var()
            sigma_square, obj = find_var_sum_cauchy(var_coeff, pcost_coeff, pcost)
        elif choice == 'maxvar':
            coeff, A, b = self.get_coeff_maxvar()
            sigma_square, obj = find_var_max(coeff, A, b, pcost)
        else:
            print("Invalid choice!")
            return 0
        self.sigma_square = sigma_square
        for i, noise_level in enumerate(sigma_square):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            res_mech.input_noise_level(noise_level)

        # print("pcost: ", pcost_coeff)
        # print("var: ", var_coeff)
        # print("sigma square: ", sigma_square)
        # print("dict: ", self.var_coeff)
        return obj

    def measurement(self):
        print("Start Measurement, total number of cells: ", len(self.res_dict))
        for i, att in enumerate(self.res_dict.keys()):
            if i % 10_000 == 0 and i > 0:
                print("Measuring cell: ", i)
            res_mech = self.res_dict[att]
            cols = [self.col_names[idx] for idx in att]
            sub_data = self.data.loc[:, cols]
            res_mech.input_data(sub_data, cols)
            res_mech.measure()

    def reconstruction(self):
        print("Start Reconstruction, total number of marginals: ", len(self.mech_dict))
        for i, att in enumerate(self.mech_dict.keys()):
            if i % 10_000 == 0 and i > 0:
                print("Reconstructing marginal: ", i)
            mech = self.mech_dict[att]
            att_subsets = all_subsets(att)
            noisy_answer = 0.0
            # todo: move it to class Mechanism
            true_answer = 0.0
            for subset in att_subsets:
                res_mech = self.res_dict[subset]
                mat_list = []

                for at in att:
                    if at in subset:
                        sub_mat = subtract_matrix(self.domains[at], False)
                        sub_pinv = np.linalg.pinv(sub_mat)
                        mat_list.append(sub_pinv)
                    else:
                        one_mat = np.ones([self.domains[at], 1]) / self.domains[at]
                        mat_list.append(one_mat)

                recon_answer = res_mech.get_recon_answer(mat_list)
                noisy_answer += recon_answer

                recon_true = res_mech.get_origin_answer(mat_list)
                true_answer += recon_true
            mech.input_noisy_answer(noisy_answer)
            mech.input_true_answer(true_answer)

    def reconstruct_covariance(self):
        for att in self.mech_dict.keys():
            mech = self.mech_dict[att]
            att_subsets = all_subsets(att)
            variance = 0.0
            for subset in att_subsets:
                res_mech = self.res_dict[subset]
                var_coeff = self.var_coeff[att, subset]
                sigma_square = res_mech.output_noise_level()
                variance += var_coeff * sigma_square
            mech.input_variance(variance)

    def get_max_variance(self):
        max_var = -np.float("inf")
        for att in self.mech_dict.keys():
            mech = self.mech_dict[att]
            var = mech.output_variance()
            print(att, var)
            max_var = max(max_var, var)
        return max_var

    def get_mean_error(self, ord=1):
        error_list = []
        N = len(self.data)
        for att in self.mech_dict:
            mech = self.mech_dict[att]
            noisy_answer = mech.get_noisy_answer()
            true_answer = mech.get_true_answer()
            l_error = np.linalg.norm(noisy_answer - true_answer, ord=ord)
            error_list.append(l_error / N)
        mean_error = np.mean(error_list)
        return mean_error


def test_Adult():
    domains = [85, 9, 100, 16, 7, 15, 6, 5, 2, 100, 100, 99, 42, 2]
    col_names = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country', 'income>50K']
    system = ResidualPlanner(domains)
    data = pd.read_csv("adult.csv")
    system.input_data(data, col_names)
    print("Len of adult dataset: ", len(data))

    att = tuple(range(len(domains)))
    total = 0
    for i in range(3, 4):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset, var_bound=1)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
    print("Total num of queries: ", total, "\n")
    return system, total


if __name__ == '__main__':
    start = time.time()
    ep_ls = [0.03, 0.1, 0.31, 1.0, 3.16, 10]

    for eps in ep_ls:
        print("------------------- ep: ", eps, "------------------")
        delta = 1e-9
        rho = cdp_rho(eps, delta)
        pcost = rho * 2

        system, total = test_Adult()
        sum_var = system.selection(choice="sumvar", pcost=pcost)
        system.measurement()
        system.reconstruction()
        l_error = system.get_mean_error(ord=1)
        print("Mean Error: ", l_error)

    end = time.time()



