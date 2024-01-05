from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from .smoothing import *


def train_image_classifier(model, train_data, hparams):
    batch_size = hparams['batch_size_training']
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               worker_init_fn=seed_worker,
                                               num_workers=1)

    if 'early_stopping' in hparams:
        early_stopping = hparams['early_stopping']
    else:
        early_stopping = np.inf

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hparams["lr"],
                                momentum=hparams["momentum"],
                                weight_decay=hparams["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    best_acc = -np.inf
    best_epoch = 0
    best_state = {}

    for epoch in tqdm(range(hparams["max_epochs"])):
        model.train()
        loss_train = 0
        correct = 0
        total = 0
        for (input, y) in tqdm(train_loader):

            if hparams['protected']:
                inputs = []
                # smooth each image during training
                for i in range(input.shape[0]):
                    img = input[i].clone().squeeze()
                    inputs.append(smooth_image(img, hparams).squeeze())
                input = torch.stack(inputs)

            x, y = input.to(hparams["device"]), y.to(hparams["device"])
            optimizer.zero_grad()
            logits = model(x)
            correct += (logits.argmax(1) == y).sum()
            total += len(y)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            loss_train += loss
            optimizer.step()

        loss_train /= len(train_loader)
        acc_train = correct/total

        if hparams['lr_scheduler']:
            if hparams['lr_scheduler'] == 'cosine':
                scheduler.step()
            else:
                scheduler.step(loss_train)

        if acc_train > best_acc:
            best_acc = acc_train
            best_epoch = epoch
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}

            if hparams["logging"]:
                print(f'Epoch {epoch:4}: '
                      f'loss_train: {loss_train.item():.5f}, '
                      f'acc_train: {acc_train.item():.5f} ')

        if epoch - best_epoch > early_stopping:
            if hparams["logging"]:
                print(f"early stopping at epoch {epoch}")
            break

    if hparams["logging"]:
        print('best_epoch', best_epoch)
    model.load_state_dict(best_state)
    return model.eval()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
