import numpy as np
import itertools


def subtract_matrix(k):
    """Return matrix C_k."""
    mat = np.zeros([k-1, k])
    for i in range(k-1):
        mat[i, i] = 1
        mat[i, i+1] = -1
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
        pass

    def output_bound(self):
        return self.var_bound

    pass


class ResMech:

    def __init__(self, domains, att):
        self.domains = domains
        self.num_att = len(self.domains)
        self.att = att

        self.noise_level = None
        self.covar = None
        self.calculated = False
        pass

    def get_query_matrix(self):
        att_set = set(list(self.att))
        mat_left = 1
        # mat_right = None
        for i in range(0, self.num_att):
            att_size = self.domains[i]
            if i in att_set:
                mat_right = subtract_matrix(att_size)
            else:
                mat_right = np.ones([1, att_size])
            mat_left = np.kron(mat_left, mat_right)
        return mat_left

    def get_core_matrix(self):
        att_set = set(list(self.att))
        mat_left = np.ones([1, 1])
        for i in range(0, self.num_att):
            att_size = self.domains[i]
            if i in att_set:
                mat_right = subtract_matrix(att_size)
                mat_left = np.kron(mat_left, mat_right)
        return mat_left

    def input_noise_level(self, noise_level):
        self.noise_level = noise_level
        self.calculated = True

    def output_noise_level(self):
        return self.noise_level

    def is_calculated(self):
        return self.calculated

    pass


