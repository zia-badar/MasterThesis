import sys
from sys import maxsize

import numpy as np
import torch


def clean_tensor_str(tensor):
    return str(np.round(tensor.cpu().numpy(), 2))

def kl_divergence(cov1, u1, cov2, u2):
    return (0.5*(torch.log(torch.det(cov2)/torch.det(cov1)) - cov1.shape[0] + torch.trace(torch.inverse(cov2)@cov1) + (u2 - u1) @ torch.inverse(cov2) @ (u2 - u1))).item()

def area_under_curve(l):
    values = torch.tensor(l)
    result = torch.sum(0.5 * torch.abs(values[1:] - values[:-1])) + torch.sum(torch.min(torch.stack([values[1:], values[:-1]]), dim=0).values)
    return np.round(result.item(), 2)

class TrainingResult:
    def __init__(self, config, starting_roc):
        self.dataset = config['dataset']
        self.class_index = config['class']
        self.starting_roc = starting_roc

        self.min_dit = maxsize
        self.min_dit_model = None
        self.min_dit_roc = None

        # self.max_dit = -1

        self.min_var = maxsize
        self.min_var_model = None
        self.min_var_roc = None

        self.max_condition_no = -1
        self.max_condition_no_roc = None
        self.max_condition_no_model = None

        self.min_condition_no = maxsize
        self.min_condition_no_roc = None
        self.min_condition_no_model = None
        self.min_condition_no_mean = None
        #
        # self.min_kl_1 = maxsize
        # self.min_kl_1_roc = None
        # self.kl_1_list = []
        # self.min_kl_2 = maxsize
        # self.min_kl_2_roc = None
        # self.kl_2_list = []
        #
        self.max_mean = -1
        self.mean_diff_list = []
        self.mean_list = []
        self.roc_list = []
        #
        # self.min_non_diag_sum = maxsize
        # self.min_non_diag_sum_roc = None
        # self.min_non_diag_sum_2 = maxsize
        # self.min_non_diag_sum_roc_2 = None
        # self.min_non_diag_sum_model_2 = None
        #
        # self.min_cov_sum = maxsize
        # self.min_cov_sum_roc = None
        # self.min_cov_sum_2 = maxsize
        # self.min_cov_sum_roc_2 = None

        self.latest_model = None
        self.latest_roc = None

        self.best_roc = torch.tensor([-1, -1, -1])
        self.condition_no_list = []
        # self.em_dist_list = []

        #logs
        self.eig_max = -1
        # self.eig_min = maxsize

    def model_state_dict(model):
        state_dict = model.state_dict().copy()
        for k, v in state_dict.items():
            state_dict[k] = v.detach().cpu()
        return state_dict

    def update(self, cov, roc, eig_val, eig_vec, e, previous_mean, current_mean):
        dit = torch.prod(torch.real(eig_val)).item()
        condition_no = torch.max(torch.real(eig_val)).item() / torch.min(torch.real(eig_val)).item()

        self.condition_no_list.append(condition_no)
        # self.em_dist_list.append(em_dist)

        if dit < self.min_dit:
            self.min_dit = dit
            self.min_dit_model = TrainingResult.model_state_dict(e)
            self.min_dit_roc = roc

        # if dit > self.max_dit:
        #     self.max_dit = dit
        #
        _mean = torch.norm(current_mean).item()
        if self.max_mean < _mean:
            self.max_mean = _mean

        self.mean_list.append(_mean)

        mean_diff = torch.norm(current_mean - previous_mean).item()
        self.mean_diff_list.append(mean_diff)

        self.roc_list.append(roc)
        #
        # kl1 = kl_divergence(torch.eye(cov.shape[0]).cuda(), torch.zeros(cov.shape[0]).cuda(), cov, mean)
        # kl2 = kl_divergence(cov, mean, torch.eye(cov.shape[0]).cuda(), torch.zeros(cov.shape[0]).cuda())
        # self.kl_1_list.append(kl1)
        # self.kl_2_list.append(kl2)
        # print(f'kl1: {kl1}, kl2: {kl2}')
        #
        # if kl1 < self.min_kl_1:
        #     self.min_kl_1 = kl1
        #     self.min_kl_1_roc = roc
        # if kl2 < self.min_kl_2:
        #     self.min_kl_2 = kl2
        #     self.min_kl_2_roc = roc

        # if var < self.min_var:
        #     self.min_var = var
        #     self.min_var_model = TrainingResult.model_state_dict(e)
        #     self.min_var_roc = roc

        if condition_no > self.max_condition_no:
            self.max_condition_no = condition_no
            self.max_condition_no_model = TrainingResult.model_state_dict(e)
            self.max_condition_no_roc = roc

        if condition_no < self.min_condition_no:
            self.min_condition_no = condition_no
            self.min_condition_no_model = TrainingResult.model_state_dict(e)
            self.min_condition_no_roc = roc
            self.min_condition_no_mean = torch.norm(current_mean)

        # if non_diag_sum < self.min_non_diag_sum:
        #     self.min_non_diag_sum = non_diag_sum
        #     self.min_non_diag_sum_roc = roc
        # if non_diag_sum_2 < self.min_non_diag_sum_2:
        #     self.min_non_diag_sum_2 = non_diag_sum_2
        #     self.min_non_diag_sum_roc_2 = roc
        #     self.min_non_diag_sum_model_2 = TrainingResult.model_state_dict(e)
        # if cov_sum < self.min_cov_sum:
        #     self.min_cov_sum = cov_sum
        #     self.min_cov_sum_roc = roc
        # if cov_sum_2 < self.min_cov_sum_2:
        #     self.min_cov_sum_2 = cov_sum_2
        #     self.min_cov_sum_roc_2 = roc

        self.latest_model = TrainingResult.model_state_dict(e)
        self.latest_roc = roc

        self.best_roc = torch.max(torch.stack((self.best_roc, roc)), dim=0)[0]

        eig_max = torch.max(torch.real(eig_val)).item()
        if eig_max > self.eig_max:
            self.eig_max = eig_max

        # eig_min = torch.min(torch.real(eig_val)).item()
        # if eig_min < self.eig_min:
        #     self.eig_min = eig_min
    #

    def __str__(self):

        return f'class: {self.class_index}' \
               f'\nbest_roc: {clean_tensor_str(self.best_roc)}' \
               f'\nmax mean: {self.max_mean}' \
               f'\nmin_condition_no: {np.round(self.min_condition_no, 2)}' \
               f'\nmax_condition_no: {np.round(self.max_condition_no, 2)}' \
               f'\nmin_condition_no_roc, mean, mean_diff: {clean_tensor_str(self.min_condition_no_roc)}, {self.min_condition_no_mean}' \
               f'\narea under condition no: {area_under_curve(self.condition_no_list)}'
