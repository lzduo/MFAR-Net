import torch


def default_collate_fn(items):
    '''default collate_fn
    '''
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]
