from sys import maxsize

import numpy as np
import torch


def clean_tensor_str(tensor):
    return str(np.round(tensor.cpu().numpy(), 2))


class TrainingResult:
    def __init__(self, config, starting_roc):
        self.dataset = config['dataset']
        self.class_index = config['class']
        self.starting_roc = starting_roc

        self.min_dit = maxsize
        self.min_dit_model = None
        self.min_dit_roc = None

        self.min_var = maxsize
        self.min_var_model = None
        self.min_var_roc = None

        self.max_condition_no = -1
        self.max_condition_no_roc = None
        self.max_condition_no_model = None

        self.min_condition_no = maxsize
        self.min_condition_no_roc = None
        self.min_condition_no_model = None

        self.latest_model = None
        self.latest_roc = None

        self.best_roc = torch.tensor([-1, -1, -1])

        #logs
        self.eig_max = -1

    def model_state_dict(model):
        state_dict = model.state_dict().copy()
        for k, v in state_dict.items():
            state_dict[k] = v.detach().cpu()
        return state_dict

    def update(self, cov, var, roc, eig_val, eig_vec, e):
        dit = torch.prod(torch.real(eig_val)).item()
        condition_no = torch.max(torch.real(eig_val)).item() / torch.min(torch.real(eig_val)).item()

        if dit < self.min_dit:
            self.min_dit = dit
            self.min_dit_model = TrainingResult.model_state_dict(e)
            self.min_dit_roc = roc

        if var < self.min_var:
            self.min_var = var
            self.min_var_model = TrainingResult.model_state_dict(e)
            self.min_var_roc = roc

        if condition_no > self.max_condition_no:
            self.max_condition_no = condition_no
            self.max_condition_no_model = TrainingResult.model_state_dict(e)
            self.max_condition_no_roc = roc

        if condition_no < self.min_condition_no:
            self.min_condition_no = condition_no
            self.min_condition_no_model = TrainingResult.model_state_dict(e)
            self.min_condition_no_roc = roc

        self.latest_model = TrainingResult.model_state_dict(e)
        self.latest_roc = roc

        self.best_roc = torch.max(torch.stack((self.best_roc, roc)), dim=0)[0]

        eig_max = torch.max(torch.real(eig_val)).item()
        if eig_max > self.eig_max:
            self.eig_max = eig_max


    def __str__(self):

        return f'class: {self.class_index}' \
               f'\nmin_dit_roc: {clean_tensor_str(self.min_dit_roc)}' \
               f'\nmin_var_roc: {clean_tensor_str(self.min_var_roc)}' \
               f'\nbest_roc: {clean_tensor_str(self.best_roc)}' \
               f'\nstarting_roc: {clean_tensor_str(self.starting_roc)}' \
               f'\nmin_condition_no: {np.round(self.min_condition_no, 2)}' \
               f'\nmin_condition_no_roc: {clean_tensor_str(self.min_condition_no_roc)}' \
               f'\nmax_condition_no: {np.round(self.max_condition_no, 2)}' \
               f'\nmax_condition_no_roc: {clean_tensor_str(self.max_condition_no_roc)}'
