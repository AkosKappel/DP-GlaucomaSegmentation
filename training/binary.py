from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from training.tools import CLASS_LABELS, update_history, save_checkpoint
from utils.metrics import update_metrics
from utils.visualization import plot_results

__all__ = ['train_binary']


def train_binary(model, criterion, optimizer, epochs, device, train_loader, val_loader=None, scheduler=None,
                 scaler=None, early_stopping_patience: int = 0, save_best_model: bool = True,
                 save_interval: int = 0, log_to_wandb: bool = False, show_plots: bool = False,
                 checkpoint_dir: str = None, log_dir: str = None, target_ids: list[int] = None, threshold: float = 0.5):
    # target_ids: defines which labels are considered as positives for binary segmentation (default: [1, 2])
    # threshold: threshold for predicted probabilities (default: 0.5)

    if target_ids is None:
        target_ids = [1, 2]
    target_ids = torch.tensor(target_ids, device=device)

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

        # Training
        train_metrics = train_one_epoch(model, criterion, optimizer, device,
                                        train_loader, scaler, target_ids, threshold)
        update_history(history, train_metrics, prefix='train')

        # Validation
        if val_loader is not None:
            val_metrics = validate_one_epoch(model, criterion, device,
                                             val_loader, scaler, target_ids, threshold)
            update_history(history, val_metrics, prefix='val')

        val_loss = history['val_loss'][-1] if val_loader is not None else history['train_loss'][-1]

        # LR scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Logging
        loader = val_loader if val_loader is not None else train_loader
        log_progress(model, loader, optimizer, history, epoch, device, target_ids, threshold,
                     log_to_wandb=log_to_wandb, log_dir=log_dir, show_plot=show_plots)

        # Checkpoints
        if save_interval and epoch % save_interval == 0 and checkpoint_dir:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, filename=f'binary-model-epoch{epoch}.pth', checkpoint_dir=checkpoint_dir)

        # Early stopping
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
                }, filename='best-binary-model.pth', checkpoint_dir=checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement == early_stopping_patience:
                print(f'=> Early stopping: best validation loss at epoch {best_epoch} with metrics {best_metrics}')
                break

    return history


def train_one_epoch(model, criterion, optimizer, device, loader, scaler=None, target_ids=None, threshold=0.5):
    if target_ids is None:
        raise ValueError('target_ids must be specified for binary segmentation')

    metric_types = [target_ids.cpu().numpy().tolist()]
    model.train()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Training')
    mean_metrics = None

    # Training loop
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.float().to(device)
        masks = masks.long().to(device)

        # Set target_ids to 1 and all other labels to 0
        masks = torch.where(torch.isin(masks, target_ids), torch.ones_like(masks), torch.zeros_like(masks))

        if scaler:
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Convert logits to probabilities
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).squeeze(1).long()

        # calculate metrics
        update_metrics(masks, preds, history, metric_types)
        history['loss'].append(loss.item())

        # display average metrics in progress bar
        mean_metrics = {k: np.mean(v) for k, v in history.items()}
        loop.set_postfix(**mean_metrics)

    return mean_metrics


def validate_one_epoch(model, criterion, device, loader, scaler=None, target_ids=None, threshold=0.5):
    if target_ids is None:
        raise ValueError('target_ids must be specified for binary segmentation')

    metric_types = [target_ids.cpu().numpy().tolist()]
    model.eval()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Validation')
    mean_metrics = None

    # Validation loop
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            # Load and prepare data
            images = images.float().to(device)
            masks = masks.long().to(device)
            masks = torch.where(torch.isin(masks, target_ids), torch.ones_like(masks), torch.zeros_like(masks))

            # Forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).squeeze(1).long()

            # calculate metrics
            update_metrics(masks, preds, history, metric_types)
            history['loss'].append(loss.item())

            # display average metrics in progress bar
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def log_progress(model, loader, optimizer, history, epoch, device, target_ids=None, threshold: float = 0.5,
                 part: str = 'validation', log_dir: str = '.', log_to_wandb: bool = False, show_plot: bool = False):
    if target_ids is None:
        raise ValueError('target_ids must be specified for binary segmentation')

    cover = 'OC cover' if target_ids.cpu().numpy().tolist() == [2] else 'OD cover'

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images, masks = batch

        images = images.float().to(device)
        masks = masks.long().to(device)

        masks = torch.where(torch.isin(masks, target_ids), torch.ones_like(masks), torch.zeros_like(masks))

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).squeeze(1).long()

        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()

    file = f'{log_dir}/epoch{epoch}.png'
    plot_results(images, masks, preds, save_path=file, show=show_plot,
                 types=['image', 'mask', 'prediction', cover])

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
