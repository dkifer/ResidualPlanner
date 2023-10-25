import numpy as np
from utils import all_subsets, Mechanism, ResMech
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import cvxpy as cp


"""
domains = [3, 3, 3 ,3 ,3]
5 attributes, each attribute 3 values
-------------------------------------------
att = ()        --> sum query
att = (0)       --> marginal A
att = (2, 3)    --> marginal CD
att = (0, 1 ,2) --> marginal ABC
"""


def find_var_sum_cauchy(var, pcost):
    """Solve the sum-of-variance optimization problem.

    min sum(var @ 1/x)
    s.t. sum(pcost @ x) == 1
    """
    T = np.sum(np.sqrt(var*pcost))**2
    x = np.sqrt(pcost / var * T)
    return x, T


def find_var_sum_cvxpy(var, pcost):
    """Solve the small optimization problem.

    min sum(1/x)
    s.t. A x <= b
    """
    size = len(var)
    x = cp.Variable(size)
    constraints = [x >= 0]
    constraints += [pcost @ cp.inv_pos(x) <= 1]
    obj = cp.Minimize(cp.sum(var @ x))
    prob = cp.Problem(obj,
                      constraints)
    prob.solve()
    print("obj sum of var:", obj.value)
    return x.value, obj.value


def find_var_sum_gurobi(var, pcost):
    """Solve the small optimization problem.

    min var @ x
    s.t. pcost @ (1/x) <= 1
    """
    env = gp.Env(empty=True)
    # env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.TIME_LIMIT = 10
    m.setParam('NonConvex', 2)
    # m.setParam(GRB.Param.OutputFlag, 0)
    size = len(var)
    x = m.addMVar(size, lb=1e-5)
    x_inv = m.addMVar(size, lb=1e-5)

    m.setObjective(var @ x, sense=GRB.MINIMIZE)
    m.addConstr(pcost @ x_inv <= 1)
    for i in range(size):
        m.addConstr(x[i] * x_inv[i] == 1.0)
    m.optimize()
    return x.X, var @ x.X


class RPSumVar:

    def __init__(self, domains):
        self.domains = domains
        self.num_of_mech = 0
        self.num_of_res = 0

        self.mech_index = {}
        self.res_index = {}
        self.id2res = {}

        self.mech_dict = {}
        self.res_dict = {}
        self.pcost_coeff = {}
        self.var_coeff = {}
        self.var_bound = {}
        self.var_coeff_sum = defaultdict(int)

        self.sparse_row = {}
        self.sparse_col = {}
        pass

    def input_mech(self, att, var_bound=1):
        mech = Mechanism(self.domains, att, var_bound)
        self.mech_dict[att] = mech
        self.mech_index[att] = self.num_of_mech
        self.num_of_mech += 1

        cur_domains = [self.domains[at] for at in att]

        att_subsets = all_subsets(att)
        for subset in att_subsets:
            if subset not in self.res_dict:
                sub_domains = [self.domains[at] for at in subset]
                pcost_coeff = np.multiply.reduce([(c - 1) / c for c in sub_domains])
                self.pcost_coeff[subset] = pcost_coeff
                res_mech = ResMech(self.domains, subset)
                self.res_dict[subset] = res_mech
                self.res_index[subset] = self.num_of_res
                self.id2res[self.num_of_res] = subset
                self.num_of_res += 1

        # be careful of integer overflow
        num_res_queries = np.multiply.reduce([c+0.0 for c in cur_domains])
        for subset in att_subsets:
            sub_domains = [self.domains[at] for at in subset]
            # be careful of the numerical overflow
            var_coeff = np.multiply.reduce([(c-1.0)/c for c in sub_domains])
            div_list = []
            for c in att:
                if c not in subset:
                    div_list.append(1.0/self.domains[c]**2)
            divisor = np.multiply.reduce(div_list)

            self.var_coeff_sum[subset] += var_coeff * divisor * num_res_queries
            if var_coeff * divisor * num_res_queries < 0:
                print(subset, var_coeff, divisor, num_res_queries)
                print("var<0: ", var_coeff * divisor * num_res_queries)

    def input_data(self, data):
        pass

    def get_residual(self):
        pass

    def output_coeff_sum(self):
        pcost_coeff_list = []
        var_sum_list = []
        for res in self.res_index.keys():
            pcost_coeff_list.append(self.pcost_coeff[res])
            var_sum_list.append(self.var_coeff_sum[res])
        var_sum = np.array(var_sum_list)
        pcost = np.array(pcost_coeff_list)
        return var_sum, pcost

    def get_noise_level(self):
        var_sum, pcost = self.output_coeff_sum()
        x_sum, obj = find_var_sum_cauchy(var_sum, pcost)
        self.var = x_sum
        if x_sum is None:
            print("failed to find a solution")
            return 0
        for i, noise_level in enumerate(x_sum):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            res_mech.input_noise_level(noise_level)
        return obj

