import torch
from tqdm.auto import tqdm


def predict_unprotected_images(hparams, model, data):
    """Standard inference of unsmoothed image classifier.

    Args:
        hparams (dict): Experiment hyperparameters.
        model (nn.Module): Image classifier.
        data (torch.utils.data.Dataset): Image classification dataset.

    Returns:
        float: Test set accuracy.
    """
    batch_size = hparams["batch_size_inference"]
    test_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    predictions = []
    with torch.no_grad():
        for (inputs, targets) in tqdm(test_loader):
            x, y = inputs.to(hparams["device"]), targets.to(hparams["device"])
            predictions.append((model(x).argmax(1) == y))

    unprotected_accuracy = torch.cat(predictions).float().mean()
    return unprotected_accuracy
