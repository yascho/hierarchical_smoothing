import torch
import torch.nn.functional as F

import numpy as np
from torch.utils.data import DataLoader
from .smoothing import smooth_graph
from tqdm.auto import tqdm
import logging


def train_gnn_inductive(model, data, hparams):

    data_training, data_inference, idx_train, idx_valid = data

    if 'early_stopping' in hparams:
        early_stopping = hparams['early_stopping']
    else:
        early_stopping = np.inf

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],
                                 weight_decay=hparams["weight_decay"])

    best_loss = np.inf
    best_epoch = 0
    best_state = {}

    for epoch in tqdm(range(hparams["max_epochs"])):
        model.train()
        optimizer.zero_grad()
        loss_train = loss(hparams, model, data_training, idx_train)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_val = loss(hparams, model, data_inference, idx_valid)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}

            if hparams["logging"]:
                logging.info(f'Epoch {epoch:4}: '
                             f'loss_train: {loss_train.item():.5f}, '
                             f'loss_val: {loss_val.item():.5f} ')

        if epoch - best_epoch > early_stopping:
            logging.info(f"early stopping at epoch {epoch}")
            break

    if hparams["logging"]:
        print('best_epoch', best_epoch)
    model.load_state_dict(best_state)
    return model.eval()


def loss(hparams, model, data, idx):
    x, edge_idx = data.x, data.edge_index
    if hparams['protected']:
        x_clean, x_noised = smooth_graph(model, x.clone(), hparams)
    else:
        x_clean, x_noised = x, x
    logits = model(x_clean, x_noised, edge_idx)
    return F.cross_entropy(logits[idx], data.y)
