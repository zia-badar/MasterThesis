import torch
from torch.nn.functional import normalize, one_hot


to do
def kmean_coreset(dataloader, model, ratio=0.01):
    nonlocal config

    k = (int)(len(dataloader.dataset) * ratio)
    centers = torch.normal(0, 1, size=(config.shift_count, k, config.encoded_dim))

    with torch.no_grad():
        model.eval()
        for _ in range(100):
            sum = torch.zeros_like(centers)
            n = torch.zeros(sum.shape[:-1])
            # for (x, _) in dataloader:
            #     x = x.cuda()
            #     x = x.reshape(-1, config.height, config.width)
            #     z, _ = model(x)
            #     z = normalize(z)
            #     z = z.reshape(config.batch_size, config.shift_count, (1+config.align_count), config.encoding_dim).transpose(1, 0, 2, 3).reshape(config.shift_count, -1, config.encoding_dim)
            #     distance = centers @ z.transpose(0, 2, 1)
            #     indexes = torch.argwhere(one_hot(torch.argmax(distance, dim=1), num_classes=k))
            #     sum[indexes[:, 0], indexes[:, 2]] += z.reshape(-1, config.encoding_dim)
            #     n[indexes[:, 0], indexes[:, 2]] += 1



            centers = sum / n[:, None]

    return centers