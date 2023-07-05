from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from training.tools import CLASS_LABELS, update_history, save_checkpoint
from utils.metrics import update_metrics
from utils.visualization import plot_results

__all__ = ['train_multiclass']


def train_multiclass(model, criterion, optimizer, epochs, device, train_loader, val_loader=None, scheduler=None,
                     scaler=None, early_stopping_patience: int = 0, save_best_model: bool = True,
                     save_interval: int = 0, log_to_wandb: bool = False, show_plots: bool = False,
                     checkpoint_dir: str = '.', log_dir: str = '.'):
    # model: model to train
    # criterion: loss function
    # optimizer: optimizer for gradient descent
    # epochs: number of epochs to train
    # device: 'cuda' or 'cpu'
    # train_loader: data loader for training set
    # val_loader: data loader for validation set
    # scheduler: learning rate scheduler (Optional)
    # scaler: scaler for mixed precision training (Optional)
    # early_stopping_patience: number of epochs to wait before stopping training if the loss does not improve
    #                          (0 or None to disable early stopping)
    # save_best_model: save the model with the best validation loss (True or False)
    # save_interval: save the model every few epochs (0 or None to disable)
    # log_to_wandb: log progress to Weights & Biases (True or False)
    # show_plots: show examples from validation set (True or False)
    # checkpoint_dir: directory to save checkpoints (default: current directory)
    # log_dir: directory to save logs (default: current directory)
    # returns: history of training and validation metrics as a dictionary of lists

    history = defaultdict(list)
    best_loss = np.inf
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0

    model = model.to(device)
    if log_to_wandb:
        wandb.watch(model, criterion)

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')

        # training
        train_metrics = train_one_epoch(model, criterion, optimizer, device, train_loader, scaler)
        update_history(history, train_metrics, prefix='train')

        # skip validation part if data loader was not provided
        if val_loader is not None:
            # validation
            val_metrics = validate_one_epoch(model, criterion, device, val_loader, scaler)
            update_history(history, val_metrics, prefix='val')

        val_loss = history['val_loss'][-1] if val_loader is not None else history['train_loss'][-1]

        # learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # log metrics locally and to wandb
        loader = val_loader if val_loader is not None else train_loader
        log_progress(model, loader, optimizer, history, epoch, device,
                     log_to_wandb=log_to_wandb, log_dir=log_dir, show_plot=show_plots)

        # save checkpoint after every few epochs
        if save_interval and epoch % save_interval == 0 and checkpoint_dir:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, filename=f'multiclass-model-epoch{epoch}.pth', checkpoint_dir=checkpoint_dir)

        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_metrics = {k: v[-1] for k, v in history.items()}
            best_epoch = epoch
            epochs_without_improvement = 0

            if save_best_model and checkpoint_dir:
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'history': history,
                }, filename='best-multiclass-model.pth', checkpoint_dir=checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement == early_stopping_patience:
                print(f'=> Early stopping: best validation loss at epoch {best_epoch} with metrics {best_metrics}')
                break

    return history


def train_one_epoch(model, criterion, optimizer, device, loader, scaler=None):
    model.train()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Training')
    mean_metrics = None

    # iterate once over all the batches in the training data loader
    for batch_idx, (images, masks) in enumerate(loop):
        # move data to device
        images = images.float().to(device)
        masks = masks.long().to(device)

        if scaler:
            # forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            # backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # convert logits to predictions
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        # calculate metrics
        update_metrics(masks, preds, history, [[1, 2], [2]])
        history['loss'].append(loss.item())

        # display average metrics in progress bar
        mean_metrics = {k: np.mean(v) for k, v in history.items()}
        loop.set_postfix(**mean_metrics)

    return mean_metrics


def validate_one_epoch(model, criterion, device, loader, scaler=None):
    model.eval()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Validation')
    mean_metrics = None

    # disable gradient calculation
    with torch.no_grad():
        # iterate once over all batches in the validation dataset
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.float().to(device)
            masks = masks.long().to(device)

            # forward pass only, no gradient calculation
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)  # model returns logits
                loss = criterion(outputs, masks)

            # convert logits to predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # calculate metrics
            update_metrics(masks, preds, history, [[1, 2], [2]])
            history['loss'].append(loss.item())

            # show summary of metrics in progress bar
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def log_progress(model, loader, optimizer, history, epoch, device, part: str = 'validation', log_dir: str = '.',
                 log_to_wandb: bool = False, show_plot: bool = False):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images, masks = batch

        images = images.float().to(device)
        masks = masks.long().to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()

    file = f'{log_dir}/epoch{epoch}.png'
    plot_results(images, masks, preds, save_path=file, show=show_plot,
                 types=['image', 'mask', 'prediction', 'OD cover', 'OC cover'])

    if log_to_wandb:
        # Log plot with example predictions
        wandb.log({f'Plotted results ({part})': wandb.Image(file)}, step=epoch)

        # Log performance metrics
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
        wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)

        # Log interactive segmentation results
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
