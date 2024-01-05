import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sparse_smoothing.utils import sparse_perturb
import torch.nn as nn


def smooth_graph_classifier(hparams, model, data, n_samples,
                            progress_bar=True):
    """ Computes the prediction of the smoothed classifier by
        sampling graphs from the smoothing distribution and classifying
        all graphs using the GNN model.

    Args:
        hparams (dict): Experiment hyperparameters.
        model (nn.Module): GNN model.
        data (torch_geometric.data.Data): The graph the GNN operates on.
        n_samples (int): Number of Monte Carlo samples (graphs) to sample.
        progress_bar (bool, optional): Whether you want a progress bar.
          Defaults to True.

    Returns:
        torch.tensor: Votes per class for each node.
    """
    nc = hparams['out_channels']
    votes = torch.zeros(data.x.shape[0], nc, dtype=torch.int32)

    with torch.no_grad():
        for sample in tqdm(range(n_samples), disable=not progress_bar):
            x, edge_idx = data.x, data.edge_index
            x_clean, x_noised = smooth_graph(model, x.clone(), hparams)
            predictions = model(x_clean, x_noised, edge_idx).argmax(1).cpu()
            votes += F.one_hot(predictions, int(nc))
    return votes


def smooth_graph(model, X, hparams):
    """Samples smoothed feature matrix depending on the smoothing distribution.

    Args:
        model (nn.Module): GNN model.
        X (torch.tensor): Feature matrix of the graph.
        hparams (dict): Experiment hyperparameters.

    Raises:
        Exception: If the smoothing distribution is not implemented.

    Returns:
        tuple: Clean and smoothed feature matrices.
    """
    smoothing_config = hparams['smoothing_config']

    X_clean = X.clone()
    if hparams['with_skip'] and smoothing_config['append_indicator']:
        zeros = torch.zeros(X.shape[0]).to(X.device).reshape(-1, 1)
        X_clean = torch.cat((X_clean, zeros), dim=1)

    smoothing_distribution = smoothing_config["smoothing_distribution"]

    if smoothing_distribution in ["sparse", "hierarchical_sparse"]:
        p = smoothing_config["p"]  # 1 for standard sparse smoothing
        p_plus = smoothing_config["p_plus"]
        p_minus = smoothing_config["p_minus"]
        if smoothing_distribution == "sparse":
            assert not smoothing_config["append_indicator"]
        i = smoothing_config["append_indicator"]  # false for sparse smoothing
        X_new = add_sparse_noise(X, p, p_plus, p_minus, append_indicator=i)
    elif smoothing_distribution == "gaussian":
        std = smoothing_config["std"]
        X_new = add_full_gaussian_noise(X, std)
    elif smoothing_distribution == "hierarchical_gaussian":
        std = smoothing_config["std"]
        p = smoothing_config["p"]
        i = smoothing_config["append_indicator"]
        X_new = add_partial_gaussian_noise(X, p, std, append_indicator=i)
    elif smoothing_distribution == "ablation":
        p = smoothing_config["p"]
        X_new = ablate(model, X, p)
    else:
        raise Exception("Not Implemented")

    return X_clean, X_new


def ablate(model, X, p):
    """Ablation smoothing (see Scholten et al., 2023).
    Ablates entire nodes by hiding their features.

    Args:
        model (nn.Module): GNN model.
        X (torch.tensor): Matrix to smooth (e.g. feature matrix of a graph).
        p (float): Ablation probability.

    Returns:
        torch.tensor: Smoothed matrix.
    """
    i = torch.tensor(np.random.binomial(1, p, X.shape[0]), device=X.device)
    X[torch.where(i)[0], :] = model.token.repeat((i.sum(), 1))
    return X


def add_partial_gaussian_noise(X, p, std, append_indicator):
    """Hierarchical smoothing with Gaussian smoothing,
       e.g. for graphs with continuous features. Adds full gaussian noise
       on a subset of rows of the matrix X.

    Args:
        X (torch.tensor): Matrix to smooth (e.g. feature matrix of a graph).
        p (float): Node selection probability.
        std (float): Standard deviation of the Gaussian distribution.
        append_indicator (bool): Whether to add the indicator column.

    Returns:
        torch.tensor: Smoothed matrix.
    """
    indicator = np.random.binomial(1, p, X.shape[0])
    i = torch.tensor(indicator, device=X.device).reshape(-1, 1)

    noised_data = add_full_gaussian_noise(X[torch.where(i)[0], :], std)
    X[torch.where(i)[0], :] = noised_data

    if append_indicator:
        X = torch.cat((X, i), dim=1)
    return X


def add_full_gaussian_noise(X, std):
    """Adds Gaussian noise on the entire tensor X.

    Args:
        X (torch.tensor): Matrix to smooth (e.g. feature matrix of a graph).
        std (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.tensor: Smoothed matrix.
    """
    return X + std * torch.randn(X.shape).to(X.device)


def add_sparse_noise(X, p, p_plus, p_minus, append_indicator):
    """Adds sparse binary noise on the matrix.

    Args:
        X (torch.tensor): Matrix to smooth (e.g. feature matrix of a graph).
        p (float): Row-selection probability p.
        p_plus (float): Probability to flip 0 -> 1.
        p_minus (float): Probability to flip 1 -> 0.
        append_indicator (bool): Whether to add the indicator column.

    Raises:
        Exception: If the smooth

    Returns:
        torch.tensor: Smoothed matrix.
    """

    indicator = np.random.binomial(1, p, X.shape[0])
    i = torch.tensor(indicator, device=X.device)
    if i.sum() > 0:
        attr_idx = X[i.bool()].nonzero().T

        try:
            per_attr_idx = sparse_perturb(data_idx=attr_idx, n=i.sum().item(),
                                          m=X.shape[1], undirected=False,
                                          pf_minus=p_minus, pf_plus=p_plus)
        except OverflowError:
            raise Exception("Perturbation too dense (p_plus too high?)")

        rows = torch.argwhere(i).T[0][per_attr_idx[0]]
        X[i.bool()] *= 0
        X[rows, per_attr_idx[1]] = 1

    if append_indicator:
        X = torch.cat((X, i.reshape(-1, 1)), dim=1)
    return X
