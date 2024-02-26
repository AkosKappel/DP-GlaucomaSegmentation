from collections import defaultdict

import numpy as np
import torch
import wandb
from tqdm import tqdm

from modules.metrics import get_best_and_worst_OC_examples, get_best_and_worst_OD_examples, update_metrics
from modules.visualization import plot_results


class DualTrainer:

    def __init__(self, model, od_criterion, oc_criterion, optimizer, device, scaler=None, od_threshold: float = 0.5,
                 oc_threshold: float = 0.5, od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0,
                 inverse_transform=None, activation=None):
        self.model = model
        self.od_criterion = od_criterion
        self.oc_criterion = oc_criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.inverse_transform = inverse_transform
        self.activation = activation or torch.sigmoid
        self.od_threshold = od_threshold
        self.oc_threshold = oc_threshold
        self.od_loss_weight = od_loss_weight
        self.oc_loss_weight = oc_loss_weight
        self.od_label = [1, 2]
        self.oc_label = [2]
        self.labels = [self.od_label, self.oc_label]

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def run_one_iteration(self, images, masks, backward: bool, history=None):
        images = images.float().to(self.device)
        masks = masks.long().to(self.device)

        # Create optic disc and optic cup binary masks
        od_masks = (masks == 1).long() + (masks == 2).long()
        oc_masks = (masks == 2).long()

        if self.scaler is None:
            # Forward pass
            od_logits, oc_logits = self.model(images)
            od_loss = self.od_criterion(od_logits, od_masks)
            oc_loss = self.oc_criterion(oc_logits, oc_masks)
            total_loss = od_loss * self.od_loss_weight + oc_loss * self.oc_loss_weight

            # Backward pass
            if backward:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        else:
            # Forward pass
            with torch.cuda.amp.autocast():
                od_logits, oc_logits = self.model(images)
                od_loss = self.od_criterion(od_logits, od_masks)
                oc_loss = self.oc_criterion(oc_logits, oc_masks)
                total_loss = od_loss * self.od_loss_weight + oc_loss * self.oc_loss_weight

            # Backward pass
            if backward:
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        # Convert logits to probabilities for OD and OC
        od_probs = self.activation(od_logits)
        od_preds = (od_probs > self.od_threshold).squeeze(1).long()

        oc_probs = self.activation(oc_logits)
        oc_preds = (oc_probs > self.oc_threshold).squeeze(1).long()

        # Join OD and OC predictions into a single tensor
        preds = torch.zeros_like(oc_preds)
        preds[od_preds == 1] = 1
        preds[oc_preds == 1] = 2

        # Inverse initial transformations like normalization, polar transform, etc.
        if self.inverse_transform is not None:
            images, masks, preds = self.inverse_transform(images, masks, preds)

        # Add new batch metrics to history
        if history is not None:
            update_metrics(masks, preds, history, self.labels)
            history['loss'].append(total_loss.item())
            history['loss_OD'].append(od_loss.item())
            history['loss_OC'].append(oc_loss.item())

        return images, masks, preds

    def train_one_epoch(self, loader):
        self.model.train()
        mean_metrics = None
        history = defaultdict(list)
        loop = tqdm(loader, total=len(loader), leave=True, desc='Training')

        for batch_idx, (images, masks) in enumerate(loop):
            self.run_one_iteration(images, masks, backward=True, history=history)
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics, learning_rate=self.get_learning_rate())

        return mean_metrics

    def validate_one_epoch(self, loader):
        self.model.eval()
        mean_metrics = None
        history = defaultdict(list)
        loop = tqdm(loader, total=len(loader), leave=True, desc='Validation')

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(loop):
                self.run_one_iteration(images, masks, backward=False, history=history)
                mean_metrics = {k: np.mean(v) for k, v in history.items()}
                loop.set_postfix(**mean_metrics)

        return mean_metrics


class DualLogger:

    def __init__(self, log_dir: str = '.', interval: int = 1, log_to_wandb: bool = False, show_plots: bool = False,
                 plot_type: str = 'all', class_labels: dict = None, num_examples: int = 4, part: str = 'validation',
                 od_threshold: float = 0.5, oc_threshold: float = 0.5):
        plot_type = plot_type.lower()
        assert plot_type in ('all', 'random', 'extreme', 'best', 'worst', 'OD', 'OC', 'none', '')
        self.dir = log_dir
        self.interval = interval
        self.wandb = log_to_wandb
        self.show = show_plots
        self.plot_type = plot_type
        self.class_labels = class_labels
        self.num_examples = num_examples
        self.part = part
        self.od_threshold = od_threshold
        self.oc_threshold = oc_threshold

    def __call__(self, model, loader, optimizer, history, epoch, device, force: bool = False):
        if self.wandb:
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
            wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)
            wandb.log({f'max_{k}': np.max(v) for k, v in history.items() if 'loss' not in k}, step=epoch)

        if not force and (not self.interval or epoch % self.interval != 0):
            return

        model.eval()
        with torch.no_grad():
            images, masks = next(iter(loader))
            images = images.float().to(device)
            masks = masks.long().to(device)

            od_masks = (masks == 1).long() + (masks == 2).long()
            oc_masks = (masks == 2).long()

            od_logits, oc_logits = model(images)
            od_probs = torch.sigmoid(od_logits)
            od_preds = (od_probs > self.od_threshold).squeeze(1).long()
            oc_probs = torch.sigmoid(oc_logits)
            oc_preds = (oc_probs > self.oc_threshold).squeeze(1).long()

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
                        'class_labels': self.class_labels,
                    },
                    'ground_truth': {
                        'mask_data': mask,
                        'class_labels': self.class_labels,
                    },
                })
                wandb.log({f'Segmentation results for OD ({self.part})': seg_img}, step=epoch)

                mask = oc_masks[i]
                pred = oc_preds[i]
                seg_img = wandb.Image(image, masks={
                    'prediction': {
                        'mask_data': pred,
                        'class_labels': self.class_labels,
                    },
                    'ground_truth': {
                        'mask_data': mask,
                        'class_labels': self.class_labels,
                    },
                })
                wandb.log({f'Segmentation results for OC ({self.part})': seg_img}, step=epoch)
                break

        if not self.dir:
            return

        od_file = f'{self.dir}/epoch{epoch}-OD.png'
        oc_file = f'{self.dir}/epoch{epoch}-OC.png'
        file_best_od = f'{self.dir}/epoch{epoch}_Best-OD.png'
        file_worst_od = f'{self.dir}/epoch{epoch}_Worst-OD.png'
        file_best_oc = f'{self.dir}/epoch{epoch}_Best-OC.png'
        file_worst_oc = f'{self.dir}/epoch{epoch}_Worst-OC.png'
        plot_types_od = ['image', 'mask', 'prediction', 'OD overlap']
        plot_types_oc = ['image', 'mask', 'prediction', 'OC overlap']

        if self.plot_type in ['all', 'random']:
            plot_results(images, od_masks, od_preds, save_path=od_file, show=self.show, types=plot_types_od)
            plot_results(images, oc_masks, oc_preds, save_path=oc_file, show=self.show, types=plot_types_oc)
            if self.wandb:
                wandb.log({f'Plotted results for OD ({self.part})': wandb.Image(od_file)}, step=epoch)
                wandb.log({f'Plotted results for OC ({self.part})': wandb.Image(oc_file)}, step=epoch)

        if self.plot_type in ['all', 'extreme', 'best', 'worst', 'OD']:
            best, worst = get_best_and_worst_OD_examples(
                model, loader, self.num_examples, device=device, thresh=self.od_threshold, out_idx=0)

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
                model, loader, self.num_examples, device=device, thresh=self.oc_threshold, out_idx=1)

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
