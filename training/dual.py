from collections import defaultdict
from IPython.display import clear_output
import numpy as np
import torch
from tqdm import tqdm
import wandb

from training.tools import CLASS_LABELS, update_history, save_checkpoint
from utils.metrics import update_metrics, get_best_and_worst_OD_examples, get_best_and_worst_OC_examples
from utils.visualization import plot_results

__all__ = ['train_dual']


def train_dual(model, od_criterion, oc_criterion, optimizer, epochs, device, train_loader, val_loader=None,
               scheduler=None, scaler=None, early_stopping_patience: int = 0, save_best_model: bool = True,
               save_interval: int = 0, log_to_wandb: bool = False, show_plots: bool = False, clear: bool = True,
               checkpoint_dir: str = '.', log_dir: str = '.', plot_examples: str = 'all', log_interval: int = 0,
               od_threshold: float = 0.5, oc_threshold: float = 0.5,
               od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0):
    # threshold1: threshold for predicted probabilities of first decoder branch (default: 0.5)
    # threshold2: threshold for predicted probabilities of second decoder branch (default: 0.5)

    history = defaultdict(list)
    best_loss = np.inf
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch = 0
    logger = DualTrainLogger(log_dir, log_interval, log_to_wandb, show_plots, plot_examples)

    model = model.to(device)
    if log_to_wandb:
        wandb.watch(model, oc_criterion)

    for epoch in range(1, epochs + 1):
        if clear and epoch % 3 == 0:
            clear_output(wait=True)

        print(f'Epoch {epoch}:')
        last_epoch = epoch

        # Training
        train_metrics = train_one_dual_epoch(model, od_criterion, oc_criterion, optimizer, device, train_loader, scaler,
                                             od_threshold, oc_threshold, od_loss_weight, oc_loss_weight)
        update_history(history, train_metrics, prefix='train')

        # Validating
        if val_loader is not None:
            val_metrics = validate_one_dual_epoch(model, od_criterion, oc_criterion, device, val_loader, scaler,
                                                  od_threshold, oc_threshold, od_loss_weight, oc_loss_weight)
            update_history(history, val_metrics, prefix='val')

        val_loss = history['val_loss'][-1] if val_loader is not None else history['train_loss'][-1]

        # LR scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Logger
        loader = val_loader if val_loader is not None else train_loader
        logger(model, loader, optimizer, history, epoch, device, od_threshold, oc_threshold)

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

    if last_epoch % log_interval != 0:
        loader = val_loader if val_loader is not None else train_loader
        logger(model, loader, optimizer, history, last_epoch, device, od_threshold, oc_threshold, force=True)

    return history


def train_one_dual_epoch(model, od_criterion, oc_criterion, optimizer, device, loader, scaler=None,
                         od_threshold: float = 0.5, oc_threshold: float = 0.5,
                         od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0):
    model.train()
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Training')
    mean_metrics = None

    # Training loop
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.float().to(device)
        masks = masks.long().to(device)

        # Create optic disc and optic cup binary masks
        od_masks = (masks == 1).long() + (masks == 2).long()
        oc_masks = (masks == 2).long()

        if scaler:
            # Forward pass
            with torch.cuda.amp.autocast():
                od_outputs, oc_outputs = model(images)
                od_loss = od_criterion(od_outputs, od_masks)
                oc_loss = oc_criterion(oc_outputs, oc_masks)
                total_loss = od_loss * od_loss_weight + oc_loss * oc_loss_weight

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            od_outputs, oc_outputs = model(images)
            od_loss = od_criterion(od_outputs, od_masks)
            oc_loss = oc_criterion(oc_outputs, oc_masks)
            total_loss = od_loss * od_loss_weight + oc_loss * oc_loss_weight

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Convert logits to probabilities for OD and OC
        od_probs = torch.sigmoid(od_outputs)
        od_preds = (od_probs > od_threshold).squeeze(1).long()

        oc_probs = torch.sigmoid(oc_outputs)
        oc_preds = (oc_probs > oc_threshold).squeeze(1).long()

        # Add new batch metrics to history
        update_metrics(masks, od_preds, history, [[1, 2]])
        update_metrics(masks, oc_preds + 1, history, [[2]])
        history['loss'].append(total_loss.item())
        history['loss_OD'].append(od_loss.item())
        history['loss_OC'].append(oc_loss.item())

        # Display mean metrics inside progress bar
        mean_metrics = {k: np.mean(v) for k, v in history.items()}
        loop.set_postfix(**mean_metrics, learning_rate=optimizer.param_groups[0]['lr'])

    return mean_metrics


def validate_one_dual_epoch(model, od_criterion, oc_criterion, device, loader, scaler=None,
                            od_threshold: float = 0.5, oc_threshold: float = 0.5,
                            od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0):
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

            od_masks = (masks == 1).long() + (masks == 2).long()
            oc_masks = (masks == 2).long()

            # Forward pass (no backprop)
            if scaler:
                with torch.cuda.amp.autocast():
                    od_outputs, oc_outputs = model(images)
                    od_loss = od_criterion(od_outputs, od_masks)
                    oc_loss = oc_criterion(oc_outputs, oc_masks)
                    total_loss = od_loss * od_loss_weight + oc_loss * oc_loss_weight
            else:
                od_outputs, oc_outputs = model(images)
                od_loss = od_criterion(od_outputs, od_masks)
                oc_loss = oc_criterion(oc_outputs, oc_masks)
                total_loss = od_loss * od_loss_weight + oc_loss * oc_loss_weight

            # Convert logits to predictions
            od_probs = torch.sigmoid(od_outputs)
            od_preds = (od_probs > od_threshold).squeeze(1).long()

            oc_probs = torch.sigmoid(oc_outputs)
            oc_preds = (oc_probs > oc_threshold).squeeze(1).long()

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


class DualTrainLogger:

    def __init__(self, log_dir: str = '.', interval: int = 1, log_to_wandb: bool = False, show_plots: bool = False,
                 plot_type: str = 'all', num_examples: int = 4, part: str = 'validation'):
        self.dir = log_dir
        self.interval = interval
        self.wandb = log_to_wandb
        self.show = show_plots
        self.plot_type = plot_type
        self.num_examples = num_examples
        self.part = part

    def __call__(self, model, loader, optimizer, history, epoch, device,
                 od_threshold: float = 0.5, oc_threshold: float = 0.5, force: bool = False):
        if self.wandb:
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
            wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)

        if not force and (not self.interval or epoch % self.interval != 0):
            return

        model.eval()
        with torch.no_grad():
            images, masks = next(iter(loader))
            images = images.float().to(device)
            masks = masks.long().to(device)

            od_masks = (masks == 1).long() + (masks == 2).long()
            oc_masks = (masks == 2).long()

            od_outputs, oc_outputs = model(images)
            od_probs = torch.sigmoid(od_outputs)
            od_preds = (od_probs > od_threshold).squeeze(1).long()
            oc_probs = torch.sigmoid(oc_outputs)
            oc_preds = (oc_probs > oc_threshold).squeeze(1).long()

            # Shift labels of optic cup from 1 to 2
            oc_masks[oc_masks == 1] = 2
            oc_preds[oc_preds == 1] = 2

            images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            od_masks = od_masks.detach().cpu().numpy()
            oc_masks = oc_masks.detach().cpu().numpy()
            od_preds = od_preds.detach().cpu().numpy()
            oc_preds = oc_preds.detach().cpu().numpy()

            images = images[:self.num_examples]
            od_masks = od_masks[:self.num_examples]
            oc_masks = oc_masks[:self.num_examples]
            od_preds = od_preds[:self.num_examples]
            oc_preds = oc_preds[:self.num_examples]

        if self.wandb:  # Log interactive segmentation results
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
                wandb.log({f'Segmentation results for OD ({self.part})': seg_img}, step=epoch)

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
                wandb.log({f'Segmentation results for OC ({self.part})': seg_img}, step=epoch)
                break

        od_file = f'{self.dir}/epoch{epoch}-OD.png'
        oc_file = f'{self.dir}/epoch{epoch}-OC.png'
        file_best_od = f'{self.dir}/epoch{epoch}_Best-OD.png'
        file_worst_od = f'{self.dir}/epoch{epoch}_Worst-OD.png'
        file_best_oc = f'{self.dir}/epoch{epoch}_Best-OC.png'
        file_worst_oc = f'{self.dir}/epoch{epoch}_Worst-OC.png'
        plot_types_od = ['image', 'mask', 'prediction', 'OD cover']
        plot_types_oc = ['image', 'mask', 'prediction', 'OC cover']
        num_examples = 4

        if self.plot_type in ['all', 'random']:
            plot_results(images, od_masks, od_preds, save_path=od_file, show=self.show, types=plot_types_od)
            plot_results(images, oc_masks, oc_preds, save_path=oc_file, show=self.show, types=plot_types_oc)
            if self.wandb:
                wandb.log({f'Plotted results for OD ({self.part})': wandb.Image(od_file)}, step=epoch)
                wandb.log({f'Plotted results for OC ({self.part})': wandb.Image(oc_file)}, step=epoch)

        if self.plot_type in ['all', 'extreme', 'best', 'worst', 'OD']:
            best, worst = get_best_and_worst_OD_examples(
                model, loader, self.num_examples, device=device, thresh=od_threshold, out_idx=0)

            b0, b1, b2 = [e[0] for e in best], [e[1] for e in best], [e[2] for e in best]
            w0, w1, w2 = [e[0] for e in worst], [e[1] for e in worst], [e[2] for e in worst]

            if self.plot_type in ['all', 'extreme', 'best', 'OD']:
                plot_results(b0, b1, b2, save_path=file_best_od, show=self.show, types=plot_types_od)
                if self.wandb:
                    wandb.log({f'Best OD examples ({self.part})': wandb.Image(file_best_od)}, step=epoch)
            if self.plot_type in ['all', 'extreme', 'worst', 'OD']:
                plot_results(w0, w1, w2, save_path=file_worst_od, show=self.show, types=plot_types_od)
                if self.wandb:
                    wandb.log({f'Worst OD examples ({self.part})': wandb.Image(file_worst_od)}, step=epoch)

        if self.plot_type in ['all', 'extreme', 'best', 'worst', 'OC']:
            best, worst = get_best_and_worst_OC_examples(
                model, loader, self.num_examples, device=device, thresh=oc_threshold, out_idx=1)

            b0, b1, b2 = [e[0] for e in best], [e[1] for e in best], [e[2] for e in best]
            w0, w1, w2 = [e[0] for e in worst], [e[1] for e in worst], [e[2] for e in worst]

            if self.plot_type in ['all', 'extreme', 'best', 'OC']:
                plot_results(b0, b1, b2, save_path=file_best_oc, show=self.show, types=plot_types_oc)
                if self.wandb:
                    wandb.log({f'Best OC examples ({self.part})': wandb.Image(file_best_oc)}, step=epoch)
            if self.plot_type in ['all', 'extreme', 'worst', 'OC']:
                plot_results(w0, w1, w2, save_path=file_worst_oc, show=self.show, types=plot_types_oc)
                if self.wandb:
                    wandb.log({f'Worst OC examples ({self.part})': wandb.Image(file_worst_oc)}, step=epoch)
