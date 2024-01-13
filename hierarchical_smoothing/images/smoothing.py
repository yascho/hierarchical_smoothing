import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sparse_smoothing.utils import sparse_perturb
import torch.nn as nn

from .models import *


def smooth_image_classifier(hparams, model, data, n_samples,
                            progress_bar=True):
    """ Computes the prediction of the smoothed classifier by
        sampling images from the smoothing distribution and classifying
        all images using the model.

    Args:
        hparams (dict): Experiment hyperparameters.
        model (nn.Module): Image classifier.
        data (torch.utils.data.Dataset): Image classification dataset.
        n_samples (int): Number of Monte Carlo samples (images) to sample.
        progress_bar (bool, optional): Whether you want a progress bar.
          Defaults to True.

    Returns:
        tuple: Votes per class for each node, and ground truth targets.
    """
    nc = hparams['out_channels']
    votes = torch.zeros(len(data), nc, dtype=torch.int32)
    test_loader = torch.utils.data.DataLoader(data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)

    targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(test_loader)):
            input = input.squeeze().to(hparams["device"])
            targets.append(target)

            batch_sizes = [hparams["batch_size_inference"]] * \
                int(n_samples/hparams["batch_size_inference"])
            if sum(batch_sizes) < n_samples:
                batch_sizes.append(n_samples - sum(batch_sizes))

            for batch_size in batch_sizes:
                x = smooth_image(input.clone(), hparams, batch_size=batch_size)
                predictions = model(x).argmax(1).cpu()
                votes[i] += F.one_hot(predictions, int(nc)).sum(0)
    return votes, targets


def smooth_image(x, hparams, batch_size=1):
    """Samples images from the smoothing distribution around x.

    Args:
        x (torch.tensor): Image data (shape: channels, width, height).
        hparams (dict): Experiment hyperparameters.
        batch_size (int, optional): Number of images to sample. Defaults to 1.
    """
    smoothing_config = hparams['smoothing_config']
    smoothing_distribution = smoothing_config["smoothing_distribution"]
    normalize = NormalizeLayer(hparams["dataset_mean"], hparams["dataset_std"])

    (num_channels, height, width) = x.shape

    if smoothing_distribution == "ablation_levine":
        k = smoothing_config["k"]
        d = smoothing_config["d"]
        # first normalize, then ablate
        x = normalize(x, batched=False)
        x = torch.cat([x, 1-x])

        batched_x = []
        for _ in range(batch_size):
            ablate = np.random.choice(np.arange(d), d-k, replace=False)
            ablated_x = x.clone()
            ablated_x.reshape(num_channels*2, -1)[:, ablate] = 0
            batched_x.append(ablated_x)
        x = torch.stack(batched_x)

    elif smoothing_distribution == "gaussian":
        std = smoothing_config["std"]
        x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x = add_full_gaussian_noise(x, std)
        x = normalize(x, batched=True)  # first add noise then normalize

    elif smoothing_distribution == "hierarchical_gaussian":
        append_indicator = smoothing_config["append_indicator"]
        std = smoothing_config["std"]
        k = smoothing_config["k"]
        d = smoothing_config["d"]
        p = 1-k/d

        indicator = np.random.binomial(1, p, batch_size*d)
        i = torch.tensor(indicator, device=x.device).reshape(
            batch_size, 1, width, height)

        x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        noised_data = add_full_gaussian_noise(x[torch.where(i)[0],
                                                :,
                                                torch.where(i)[2],
                                                torch.where(i)[3]], std)
        x[torch.where(i)[0], :, torch.where(i)[2],
          torch.where(i)[3]] = noised_data
        x = normalize(x, batched=True)  # first add noise then normalize

        if append_indicator:
            x = torch.cat((x, i), dim=1)

    return x


def add_full_gaussian_noise(X, std):
    return X + std * torch.randn(X.shape).to(X.device)
