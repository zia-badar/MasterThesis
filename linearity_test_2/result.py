from sys import maxsize

from torch.distributions import MultivariateNormal


class training_result:
    def __init__(self, projection, translation, config):
        self.projection = projection
        self.translation = translation
        self.config = config

        self.min_condition_no_model = None
        self.min_condition_no = maxsize
        self.min_condition_no_distribution = None

    def update(self, model, mean, cov, condition_no):

        if condition_no < self.min_condition_no:
            self.min_condition_no = condition_no
            self.min_condition_no_model = model.state_dict().copy()
            self.min_condition_no_distribution = MultivariateNormal(mean, cov)
