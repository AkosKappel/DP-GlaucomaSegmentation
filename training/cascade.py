from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from training.tools import CLASS_LABELS, update_history, save_checkpoint
from utils.metrics import update_metrics
from utils.visualization import plot_results

__all__ = ['train_cascade']


def train_cascade(od_model, oc_model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                  scheduler=None, scaler=None, early_stopping_patience: int = 0, save_best_model: bool = True,
                  save_interval: int = 0, log_to_wandb: bool = False, show_plots: bool = False,
                  checkpoint_dir: str = '.', log_dir: str = '.', od_threshold: float = 0.5, oc_threshold: float = 0.5):
    # od_model: pre-trained model for optic disc segmentation
    # oc_model: model for optic cup segmentation that will be trained from scratch
    # od_threshold: decides whether a predicted optic disc probability is considered as a positive sample (default: 0.5)
    # oc_threshold: decides whether a predicted optic cup probability is considered as a positive sample (default: 0.5)

    history = defaultdict(list)
    best_loss = np.inf
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0

    od_model = od_model.to(device)
    oc_model = oc_model.to(device)
    if log_to_wandb:
        wandb.watch(oc_model, criterion)

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')

        # Training
        train_metrics = train_one_epoch(od_model, oc_model, criterion, optimizer, device, train_loader, scaler,
                                        od_threshold, oc_threshold)
        update_history(history, train_metrics, prefix='train')

        # Validating
        if val_loader is not None:
            val_metrics = validate_one_epoch(od_model, oc_model, criterion, device, val_loader, scaler,
                                             od_threshold, oc_threshold)
            update_history(history, val_metrics, prefix='val')

        val_loss = history['val_loss'][-1] if val_loader is not None else history['train_loss'][-1]

        # LR scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Logging and plotting
        loader = val_loader if val_loader is not None else train_loader
        log_progress(od_model, oc_model, loader, optimizer, history, epoch, device, od_threshold, oc_threshold,
                     log_to_wandb=log_to_wandb, log_dir=log_dir, show_plot=show_plots)

        # Checkpoints saving
        if save_interval and epoch % save_interval == 0 and checkpoint_dir:
            save_checkpoint({
                'epoch': epoch,
                'model': oc_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, filename=f'cascade-model-epoch{epoch}.pth', checkpoint_dir=checkpoint_dir)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_metrics = {k: v[-1] for k, v in history.items()}
            best_epoch = epoch
            epochs_without_improvement = 0

            if save_best_model and checkpoint_dir:
                save_checkpoint({
                    'epoch': epoch,
                    'model': oc_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'history': history,
                }, filename='best-cascade-model.pth', checkpoint_dir=checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement == early_stopping_patience:
                print(f'=> Early stopping: best validation loss at epoch {best_epoch} with metrics {best_metrics}')
                break

    return history


def train_one_epoch(od_model, oc_model, criterion, optimizer, device, loader, scaler=None,
                    od_threshold: float = 0.5, oc_threshold: float = 0.5):
    od_model.eval()
    oc_model.train()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Training')
    mean_metrics = None

    # Training loop
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.float().to(device)
        masks = masks.long().to(device)

        # Apply first model to get optic disc masks which get passed to the second model
        with torch.no_grad():
            od_outputs = od_model(images)
            od_probs = torch.sigmoid(od_outputs)
            od_preds = (od_probs > od_threshold).squeeze(1).long()

        # Crop images to optic disc boundaries
        cropped_images = images * od_preds.unsqueeze(1)

        # Create optic cup only masks
        oc_masks = (masks == 2).long()

        if scaler:
            # Forward pass
            with torch.cuda.amp.autocast():
                oc_outputs = oc_model(cropped_images)
                loss = criterion(oc_outputs, oc_masks)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            oc_outputs = oc_model(cropped_images)
            loss = criterion(oc_outputs, oc_masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Convert logits to probabilities
        oc_probs = torch.sigmoid(oc_outputs)
        oc_preds = (oc_probs > oc_threshold).squeeze(1).long()

        # calculate metrics
        update_metrics(masks, oc_preds + 1, history, [[2]])
        history['loss'].append(loss.item())

        # display average metrics in progress bar
        mean_metrics = {k: np.mean(v) for k, v in history.items()}
        loop.set_postfix(**mean_metrics)

    return mean_metrics


def validate_one_epoch(od_model, oc_model, criterion, device, loader, scaler=None,
                       od_threshold: float = 0.5, oc_threshold: float = 0.5):
    od_model.eval()
    oc_model.eval()
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

            # Create optic disc masks
            od_outputs = od_model(images)
            od_probs = torch.sigmoid(od_outputs)
            od_preds = (od_probs > od_threshold).squeeze(1).long()

            # Apply masks to images
            cropped_images = images * od_preds.unsqueeze(1)

            # Create optic cup only masks
            oc_masks = (masks == 2).long()

            # Forward pass (no backprop)
            if scaler:
                with torch.cuda.amp.autocast():
                    oc_outputs = oc_model(cropped_images)
                    loss = criterion(oc_outputs, oc_masks)
            else:
                oc_outputs = oc_model(cropped_images)
                loss = criterion(oc_outputs, oc_masks)

            # Convert logits to predictions
            oc_probs = torch.sigmoid(oc_outputs)
            oc_preds = (oc_probs > oc_threshold).squeeze(1).long()

            # calculate metrics
            update_metrics(masks, oc_preds + 1, history, [[2]])
            history['loss'].append(loss.item())

            # show summary of metrics in progress bar
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def log_progress(od_model, oc_model, loader, optimizer, history, epoch, device,
                 od_threshold: float = 0.5, oc_threshold: float = 0.5,
                 part: str = 'validation', log_dir: str = '.', log_to_wandb: bool = False, show_plot: bool = False):
    od_model.eval()
    oc_model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images, masks = batch

        images = images.float().to(device)
        masks = masks.long().to(device)

        od_outputs = od_model(images)
        od_probs = torch.sigmoid(od_outputs)
        od_preds = (od_probs > od_threshold).squeeze(1).long()

        cropped_images = images * od_preds.unsqueeze(1)

        oc_masks = (masks == 2).long()

        oc_outputs = oc_model(cropped_images)
        oc_probs = torch.sigmoid(oc_outputs)
        oc_preds = (oc_probs > oc_threshold).squeeze(1).long()

        oc_masks[oc_masks == 1] = 2
        oc_preds[oc_preds == 1] = 2

        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        oc_masks = oc_masks.cpu().numpy()
        oc_preds = oc_preds.cpu().numpy()

    file = f'{log_dir}/epoch{epoch}.png'
    plot_results(images, oc_masks, oc_preds, save_path=file, show=show_plot,
                 types=['image', 'mask', 'prediction', 'OC cover'])

    if log_to_wandb:
        # Log plot with example predictions
        wandb.log({f'Plotted results ({part})': wandb.Image(file)}, step=epoch)

        # Log performance metrics
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
        wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)

        # Log interactive segmentation results
        for i, image in enumerate(images):
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
