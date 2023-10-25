import numpy as np
from utils import all_subsets, Mechanism, ResMech
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
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


def find_var_max_cvxpy(coeff, A, b):
    """Solve the small optimization problem.

    min sum(1/x)
    s.t. A x <= b
    """
    size = len(coeff)
    x = cp.Variable(size)
    constraints = [x >= 0]
    constraints += [A @ x - b <= 0]
    obj = cp.Minimize(cp.sum(coeff @ cp.inv_pos(x)))
    prob = cp.Problem(obj,
                      constraints)
    prob.solve()
    # print("obj:", obj.value)
    return x.value, obj.value


def find_var_max_gurobi(coeff, A, b, time_limit=10):
    """Solve the small optimization problem.

    min sum(1/x)
    s.t. A x <= b
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.TIME_LIMIT = time_limit
    m.setParam('NonConvex', 2)
    # m.setParam(GRB.Param.OutputFlag, 0)
    size = np.shape(A)[1]
    x = m.addMVar(size, lb=1e-5)
    x_inv = m.addMVar(size, lb=1e-5)

    m.setObjective(coeff @ x_inv, sense=GRB.MINIMIZE)
    m.addConstr(A @ x <= b)
    for i in range(size):
        m.addConstr(x[i] * x_inv[i] == 1.0)
    m.optimize()
    return x.X, m.getObjective().getValue()


class RPMaxVar:

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

        self.sparse_row = {}
        self.sparse_col = {}
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
                pcost_coeff = np.multiply.reduce([(c - 1) / c for c in sub_domains])
                self.pcost_coeff[subset] = pcost_coeff
                res_mech = ResMech(self.domains, subset)
                self.res_dict[subset] = res_mech
                self.res_index[subset] = self.num_of_res
                self.id2res[self.num_of_res] = subset
                self.num_of_res += 1

        row_list = []
        col_list = []
        var_list = []
        cur_id = self.mech_index[att]

        for subset in att_subsets:
            sub_domains = [self.domains[at] for at in subset]
            # be careful of the numerical overflow
            var_coeff = np.multiply.reduce([(c - 1) / c for c in sub_domains])
            div_list = []
            for c in att:
                if c not in subset:
                    div_list.append(1.0 / self.domains[c] ** 2)
            divisor = np.multiply.reduce(div_list)
            var_coeff = var_coeff * divisor
            if var_coeff * divisor < 0:
                print(subset, var_coeff, divisor)
                print("var<0: ", var_coeff * divisor)

            var_list.append(var_coeff)
            row_list.append(cur_id)
            sub_id = self.res_index[subset]
            col_list.append(sub_id)

        self.sparse_row[att] = np.array(row_list)
        self.sparse_col[att] = np.array(col_list)
        self.var_coeff[att] = np.array(var_list)

    def input_data(self, data):
        pass

    def get_residual(self):
        pass

    def output_mat_bound(self):
        mat_list = []
        bound_list = []

        for mech in self.mech_dict.values():
            bmat = mech.output_mat()
            m, _ = np.shape(bmat)
            bound = mech.output_bound()
            constraints = np.ones(m) * bound
            mat_list.append(bmat)
            bound_list.append(constraints)

        mat = np.concatenate(mat_list)
        bound = np.concatenate(bound_list)
        return mat, bound

    def output_coeff(self):
        pcost_coeff_list = []
        var_coeff_list = []
        row_list = []
        col_list = []
        var_bound_list = []
        for res in self.res_index.keys():
            pcost_coeff_list.append(self.pcost_coeff[res])
        for mech in self.mech_index.keys():
            var_coeff_list.append(self.var_coeff[mech])
            row_list.append(self.sparse_row[mech])
            col_list.append(self.sparse_col[mech])
            var_bound_list.append(self.var_bound[mech])
        coeff = np.array(pcost_coeff_list)
        val = np.concatenate(var_coeff_list)
        row = np.concatenate(row_list)
        col = np.concatenate(col_list)
        A = sp.csr_matrix((val, (row, col)), shape=(
            self.num_of_mech, self.num_of_res))
        b = np.array(var_bound_list)
        return coeff, A, b

    def get_noise_level(self, solver="cvxpy", time_limit=10):
        coeff, A, b = self.output_coeff()
        if solver == "cvxpy":
            var, obj = find_var_max_cvxpy(coeff, A, b)
        else:
            var, obj = find_var_max_gurobi(coeff, A, b, time_limit)
        self.var = var
        for i, noise_level in enumerate(var):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            res_mech.input_noise_level(noise_level)
        return obj


