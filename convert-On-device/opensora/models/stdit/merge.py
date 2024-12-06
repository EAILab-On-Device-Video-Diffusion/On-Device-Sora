import torch
from typing import Tuple, Callable

# for custom version, not original Video ToMe
def merge(x):
    BS, T, C = x.shape
    
    # Ensure T is even for merging
    if T % 2 != 0:
        x = x[:, :-1, :]  # Drop the last element if T is odd

    merged_x = (x[:, 0::2, :] + x[:, 1::2, :]) / 2
    return merged_x


def unmerge(x):
    BS, T, C = x.shape
    # T_new = T * merge_factor
    # x = x.unsqueeze(2).expand(BS, T, merge_factor, C).reshape(BS, T_new, C)
    unmerged_x = x.unsqueeze(2).expand(-1, -1, 2, -1)  # (BS, T//2, 2, C)
    unmerged_x = unmerged_x.contiguous().view(BS, T * 2, C)
    if T % 2 != 0 and T != 15:
        unmerged_x = torch.cat([unmerged_x, unmerged_x[:, -1:, :]], dim=1)
    return unmerged_x