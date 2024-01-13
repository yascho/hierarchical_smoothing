import torch
import random
import numpy as np


def avg_results(exp, key1, key2, seeds):
    max_r = max([exp[k][key1][key2].shape[0] for k in seeds])
    max_ra = max([exp[k][key1][key2].shape[1] for k in seeds])
    max_rd = max([exp[k][key1][key2].shape[2] for k in seeds])
    merged = np.zeros((len(seeds), max_r, max_ra, max_rd))

    for i in range(len(seeds)):
        array_i = exp[seeds[i]][key1][key2]
        merged[i,
               :array_i.shape[0],
               :array_i.shape[1],
               :array_i.shape[2]] += array_i

    return minimize(merged.mean(0)), minimize(merged.std(0))


def minimize(array: np.array):
    return array.shape, tuple(t.tolist() for t in array.nonzero()),
    array[array.nonzero()].tolist()


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
