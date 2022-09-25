import torch
from torch.nn.functional import normalize


class Loss():

    def __init__(self, dataloader, model):
        nonlocal config
        self.model = model
        self.config = config

        self.masks = []
        for (x, _) in dataloader:
            x_shape = x.size()
            mask_shape = torch.tensor([torch.prod(x_shape[:2]), torch.prod(x_shape[:2]), x_shape[2], x_shape[2]])
            if self.get_mask_shape_like(mask_shape) == None:
                pos_mask = torch.zeros_like(mask_shape)
                for i in range(mask_shape[0]):
                    pos_mask[i, i, :, :] = 1
                for i in range(mask_shape[2]):
                    pos_mask[:, :, i, i] = 0

                neg_mask = torch.ones_like(mask_shape)
                for i in range(mask_shape[0]):
                    neg_mask[i, i, :, :] = 0

                prob_mask = torch.zeros_like(torch.cat(x_shape[:4], x_shape[1]))
                for i in range(x_shape[1]):
                    prob_mask[:, i, :, i] = 1

                self.masks.append((pos_mask, neg_mask, prob_mask))

    def get_mask_shape_like(self, shape):
        for mask in self.masks:
            if torch.all(mask[0].size() == shape).item():
                return mask

        return None

    def __call__(self, batch):
        xs, _ = batch
        xs = xs.cuda()
        x_shape = xs.size()
        x_collapsed_shape = torch.stack(torch.prod(x_shape[:4]), x_shape[4:])
        zs, p_s_x = self.model(xs.rehape(x_collapsed_shape))
        zs = normalize(zs)
        zs = zs.reshape(torch.prod(x_shape[:2]), x_shape[2], -1)
        sim = torch.exp(torch.squeeze(torch.unsqueeze(zs, dim=1) @ torch.transpose(zs, 0, 2, 1), -1)/self.config.temp)
        positive_mask, negative_mask, prob_mask = self.get_mask_shape_like(sim.size())
        pos = torch.sum(sim * positive_mask, dim=(1, 3))
        neg = torch.sum(sim * negative_mask, dim=(1, 3))
        l_con_si = -torch.mean(torch.log(pos/(pos+neg)))
        p_s_x = p_s_x.reshape(torch.cat(x_shape[:4], x_shape[1]))
        l_cls_si = -torch.mean(torch.log(p_s_x * self.prob_mask))
        l_csi = l_con_si + l_cls_si

        return l_csi