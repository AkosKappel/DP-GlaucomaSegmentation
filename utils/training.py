from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from utils.checkpoint import save_checkpoint
from utils.datasets import ORIGA_MEANS, ORIGA_STDS
from utils.metrics import get_performance_metrics
from utils.visualization import plot_results

CLASS_LABELS = {
    0: 'Background',
    1: 'Optic Disc',
    2: 'Optic Cup',
}


def train_one_epoch(model, criterion, optimizer, device, loader):
    model.train()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Training')
    mean_metrics = None

    # iterate once over all the batches in the training data loader
    for batch_idx, (images, masks) in enumerate(loop):
        # move data to device
        images = images.to(device=device)
        masks = masks.to(device=device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, masks.long())

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate metrics
        preds = torch.argmax(outputs, dim=1)
        metrics = get_performance_metrics(masks.cpu(), preds.cpu())

        # update training history
        history['loss'].append(loss.item())
        for k, v in metrics.items():
            history[k].append(v)

        # display average metrics at the end of the epoch
        last_batch = batch_idx == total - 1
        if last_batch:
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def validate_one_epoch(model, criterion, device, loader):
    model.eval()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Validation')
    mean_metrics = None

    # disable gradient calculation
    with torch.no_grad():
        # iterate once over all batches in the validation dataset
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(device=device)
            masks = masks.to(device=device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.long())

            # calculate metrics
            preds = torch.argmax(outputs, dim=1)
            metrics = get_performance_metrics(masks.cpu(), preds.cpu())

            # update validation history
            history['loss'].append(loss.item())
            for k, v in metrics.items():
                history[k].append(v)

            # show summary after last batch
            last_batch = batch_idx == total - 1
            if last_batch:
                mean_metrics = {k: np.mean(v) for k, v in history.items()}
                loop.set_postfix(**mean_metrics)

    return mean_metrics


def log_progress(model, loader, optimizer, history, epoch, device, part='validation', log_dir='.', log_to_wandb=False):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images, masks = batch

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()

    if log_to_wandb:
        for i, image in enumerate(images):
            mask = masks[i]
            pred = preds[i]
            seg_img = wandb.Image(image, masks={
                'prediction': {
                    'mask_data': pred,
                    'class_labels': CLASS_LABELS,
                },
                'ground_truth': {
                    'mask_data': mask,
                    'class_labels': CLASS_LABELS,
                },
            })
            wandb.log({f'Segmentation results ({part})': seg_img}, step=epoch)
            break

    file = f'{log_dir}/epoch{epoch}.png'
    plot_results(images, masks, preds, save_path=file, show=False, mean=ORIGA_MEANS, std=ORIGA_STDS)

    if log_to_wandb:
        wandb.log({f'Plotted results ({part})': wandb.Image(file)}, step=epoch)
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
        wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)


def train(model, criterion, optimizer, epochs, device, train_loader, val_loader=None, scheduler=None,
          early_stopping_patience=0, save_best_model=True, save_interval=0, log_to_wandb=False,
          checkpoint_dir='.', log_dir='.'):
    history = defaultdict(list)
    best_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    model = model.to(device=device)
    if log_to_wandb:
        wandb.watch(model, criterion)

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')

        # training
        train_metrics = train_one_epoch(model, criterion, optimizer, device, train_loader)
        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)

        # skip validation part if data loader was not provided
        if val_loader is not None:
            # validation
            val_metrics = validate_one_epoch(model, criterion, device, val_loader)
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)

        val_loss = history['val_loss'][-1] if val_loader is not None else history['train_loss'][-1]

        # learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # log metrics locally and to wandb
        loader = val_loader if val_loader else train_loader
        log_progress(model, loader, optimizer, history, epoch, device, log_to_wandb=log_to_wandb, log_dir=log_dir)

        # save checkpoint after every few epochs
        if save_interval and epoch % save_interval == 0 and checkpoint_dir:
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'model-epoch{epoch}.pth', checkpoint_dir=checkpoint_dir)

        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            if save_best_model and checkpoint_dir:
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename='best-model.pth', checkpoint_dir=checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement == early_stopping_patience:
                print(f'Early stopping: best validation loss = {best_loss:.4f} at epoch {best_epoch}')
                break

    return history


def train_with_scaler():
    pass  # TODO