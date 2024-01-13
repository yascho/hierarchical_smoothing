def predict_unprotected_graphs(model, data, idx_test):
    """Standard inference of unsmoothed GNN.

    Args:
        model (nn.Module): GNN model.
        data (torch_geometric.data.Data): The graph the GNN operates on.
        idx_test (np.array): Indices of test nodes.

    Returns:
        float: Test set accuracy.
    """

    x, edge_idx = data.x, data.edge_index
    predictions = model(x, x, edge_idx).cpu()
    correct = predictions.argmax(1)[idx_test] == data.y.cpu()

    unprotected_accuracy = correct.float().mean()
    return unprotected_accuracy
