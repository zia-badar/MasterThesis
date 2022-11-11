class Loss():
    def __init__(self, _lambda=1e-6):
        self._lambda = _lambda

    def __call__(self, model, batch, c):
        x, l = batch
        f_x = model(x.cuda())
        return ((f_x - c)**2).sum(dim=1).mean()
        # return (1/f_x.shape[0]) * ((f_x - c)**2).sum() + (self._lambda / 2) * ((model.get_weights())**2).sum()