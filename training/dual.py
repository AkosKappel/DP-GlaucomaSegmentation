from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from training.tools import CLASS_LABELS, update_history, save_checkpoint
from utils.metrics import update_metrics
from utils.visualization import plot_results

__all__ = ['train_dual']


def train_dual(model, criterion, optimizer, epochs, device, train_loader, val_loader=None, scheduler=None,
               scaler=None, early_stopping_patience: int = 0, save_best_model: bool = True,
               save_interval: int = 0, log_to_wandb: bool = False, show_plots: bool = False,
               checkpoint_dir: str = '.', log_dir: str = '.', od_ids: list[int] = None, oc_ids: list[int] = None,
               threshold1: float = 0.5, threshold2: float = 0.5):
    # od_ids: defines which labels are considered as part of optic disc (default: [1, 2])
    # oc_ids: defines which labels are considered as part of optic cup (default: [2])
    # threshold1: threshold for predicted probabilities of first decoder branch (default: 0.5)
    # threshold2: threshold for predicted probabilities of second decoder branch (default: 0.5)

    if od_ids is None:
        od_ids = [1, 2]
    if oc_ids is None:
        oc_ids = [2]
    od_ids = torch.tensor(od_ids).to(device)
    oc_ids = torch.tensor(oc_ids).to(device)

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
        train_metrics = train_one_epoch(model, criterion, optimizer, device, train_loader,
                                        scaler, od_ids, oc_ids, threshold1, threshold2)
        update_history(history, train_metrics, prefix='train')

        # Validating
        if val_loader is not None:
            val_metrics = validate_one_epoch(model, criterion, device, val_loader, scaler,
                                             od_ids, oc_ids, threshold1, threshold2)
            update_history(history, val_metrics, prefix='val')

        val_loss = history['val_loss'][-1] if val_loader is not None else history['train_loss'][-1]

        # LR scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Logger
        loader = val_loader if val_loader is not None else train_loader
        log_progress(model, loader, optimizer, history, epoch, device, od_ids, oc_ids, threshold1, threshold2,
                     log_to_wandb=log_to_wandb, log_dir=log_dir, show_plot=show_plots)

        # Checkpoints
        if save_interval and epoch % save_interval == 0 and checkpoint_dir:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, filename=f'dual-model-epoch{epoch}.pth', checkpoint_dir=checkpoint_dir)

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
                }, filename='best-dual-model.pth', checkpoint_dir=checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement == early_stopping_patience:
                print(f'=> Early stopping: best validation loss at epoch {best_epoch} with metrics {best_metrics}')
                break

    return history


def train_one_epoch(model, criterion, optimizer, device, loader, scaler=None, od_ids=None, oc_ids=None,
                    threshold1: float = 0.5, threshold2: float = 0.5):
    assert od_ids is not None, 'od_ids must be specified for binary segmentation with dual decoder network'
    assert oc_ids is not None, 'oc_ids must be specified for binary segmentation with dual decoder network'

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
        od_masks = torch.where(torch.isin(masks, od_ids), torch.ones_like(masks), torch.zeros_like(masks))
        oc_masks = torch.where(torch.isin(masks, oc_ids), torch.ones_like(masks), torch.zeros_like(masks))

        if scaler:
            # Forward pass
            with torch.cuda.amp.autocast():
                od_outputs, oc_outputs = model(images)
                od_loss = criterion(od_outputs, od_masks)
                oc_loss = criterion(oc_outputs, oc_masks)
                total_loss = od_loss + oc_loss

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            od_outputs, oc_outputs = model(images)
            od_loss = criterion(od_outputs, od_masks)
            oc_loss = criterion(oc_outputs, oc_masks)
            total_loss = od_loss + oc_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Convert logits to probabilities for OD and OC
        od_probs = torch.sigmoid(od_outputs)
        od_preds = (od_probs > threshold1).squeeze(1).long()

        oc_probs = torch.sigmoid(oc_outputs)
        oc_preds = (oc_probs > threshold2).squeeze(1).long()

        # Add new batch metrics to history
        update_metrics(masks, od_preds, history, [[1, 2]])
        update_metrics(masks, oc_preds + 1, history, [[2]])
        history['loss'].append(total_loss.item())
        history['loss_OD'].append(od_loss.item())
        history['loss_OC'].append(oc_loss.item())

        # Display mean metrics inside progress bar
        mean_metrics = {k: np.mean(v) for k, v in history.items()}
        loop.set_postfix(**mean_metrics)

    return mean_metrics


def validate_one_epoch(model, criterion, device, loader, scaler=None, od_ids=None, oc_ids=None,
                       threshold1: float = 0.5, threshold2: float = 0.5):
    assert od_ids is not None, 'od_ids must be specified for binary segmentation with dual decoder network'
    assert oc_ids is not None, 'oc_ids must be specified for binary segmentation with dual decoder network'

    model.eval()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Validation')
    mean_metrics = None

    # Validation loop
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            # Prepare data
            images = images.float().to(device)
            masks = masks.long().to(device)

            od_masks = torch.where(torch.isin(masks, od_ids), torch.ones_like(masks), torch.zeros_like(masks))
            oc_masks = torch.where(torch.isin(masks, oc_ids), torch.ones_like(masks), torch.zeros_like(masks))

            # Forward pass (no backprop)
            if scaler:
                with torch.cuda.amp.autocast():
                    od_outputs, oc_outputs = model(images)
                    od_loss = criterion(od_outputs, od_masks)
                    oc_loss = criterion(oc_outputs, oc_masks)
                    total_loss = od_loss + oc_loss
            else:
                od_outputs, oc_outputs = model(images)
                od_loss = criterion(od_outputs, od_masks)
                oc_loss = criterion(oc_outputs, oc_masks)
                total_loss = od_loss + oc_loss

            # Convert logits to predictions
            od_probs = torch.sigmoid(od_outputs)
            od_preds = (od_probs > threshold1).squeeze(1).long()

            oc_probs = torch.sigmoid(oc_outputs)
            oc_preds = (oc_probs > threshold2).squeeze(1).long()

            # calculate metrics
            update_metrics(masks, od_preds, history, [[1, 2]])
            update_metrics(masks, oc_preds + 1, history, [[2]])
            history['loss'].append(total_loss.item())
            history['loss_OD'].append(od_loss.item())
            history['loss_OC'].append(oc_loss.item())

            # show summary of metrics in progress bar
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def log_progress(model, loader, optimizer, history, epoch, device,
                 od_ids=None, oc_ids=None, threshold1: float = 0.5, threshold2: float = 0.5,
                 part: str = 'validation', log_dir: str = '.', log_to_wandb: bool = False, show_plot: bool = False):
    assert od_ids is not None, 'od_ids must be specified for binary segmentation with dual decoder network'
    assert oc_ids is not None, 'oc_ids must be specified for binary segmentation with dual decoder network'

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images, masks = batch

        images = images.float().to(device)
        masks = masks.long().to(device)

        od_masks = torch.where(torch.isin(masks, od_ids), torch.ones_like(masks), torch.zeros_like(masks))
        oc_masks = torch.where(torch.isin(masks, oc_ids), torch.ones_like(masks), torch.zeros_like(masks))

        od_outputs, oc_outputs = model(images)
        od_probs = torch.sigmoid(od_outputs)
        od_preds = (od_probs > threshold1).squeeze(1).long()
        oc_probs = torch.sigmoid(oc_outputs)
        oc_preds = (oc_probs > threshold2).squeeze(1).long()

        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        od_masks = od_masks.cpu().numpy()
        oc_masks = oc_masks.cpu().numpy()
        od_preds = od_preds.cpu().numpy()
        oc_preds = oc_preds.cpu().numpy()

    od_file = f'{log_dir}/epoch{epoch}-OD.png'
    plot_results(images, od_masks, od_preds, save_path=od_file, show=show_plot,
                 types=['image', 'mask', 'prediction', 'OD cover'])
    oc_file = f'{log_dir}/epoch{epoch}-OC.png'
    plot_results(images, oc_masks + 1, oc_preds + 1, save_path=oc_file, show=show_plot,
                 types=['image', 'mask', 'prediction', 'OC cover'])

    if log_to_wandb:
        # Log plot with example predictions
        wandb.log({f'Plotted results for OD ({part})': wandb.Image(od_file)}, step=epoch)
        wandb.log({f'Plotted results for OC ({part})': wandb.Image(oc_file)}, step=epoch)

        # Log performance metrics
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
        wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)

        # Log interactive segmentation results
        for i, image in enumerate(images):
            mask = od_masks[i]
            pred = od_preds[i]
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
            wandb.log({f'Segmentation results for OD ({part})': seg_img}, step=epoch)

            mask = oc_masks[i]
            pred = oc_preds[i]
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
            wandb.log({f'Segmentation results for OC ({part})': seg_img}, step=epoch)
            break
