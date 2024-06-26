from collections import defaultdict

import numpy as np
import torch
import wandb
from tqdm import tqdm

from modules.metrics import get_best_and_worst_OC_examples, get_best_and_worst_OD_examples, update_metrics
from modules.visualization import plot_results


class BinaryTrainer:

    def __init__(self, model, criterion, optimizer, device, scaler=None, target_ids=None, threshold: float = 0.5,
                 inverse_transform=None, activation=None, postfix_metrics: list[str] = None):
        assert target_ids is not None, 'target_ids must be specified for binary segmentation'
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.inverse_transform = inverse_transform
        self.activation = activation or torch.sigmoid
        self.threshold = threshold
        self.target_ids = target_ids
        self.labels = [target_ids.detach().cpu().numpy().tolist()]
        self.postfix_metrics = postfix_metrics

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def run_one_iteration(self, images, masks, backward: bool, history=None):
        images = images.float().to(self.device)
        masks = masks.long().to(self.device)

        # Binarize mask by setting target ID labels to 1 and all other labels to 0 in mask
        masks = torch.where(torch.isin(masks, self.target_ids), 1, 0)

        if self.scaler is None:
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            # Backward pass
            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        else:
            # Forward pass
            with torch.cuda.amp.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            # Backward pass
            if backward:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        # Convert logits to probabilities and apply threshold to get predictions
        probs = self.activation(logits)
        preds = (probs > self.threshold).squeeze(1).long()

        # Shift predicted class labels for OC
        if self.labels == [[2]]:
            preds[preds == 1] = 2
            masks[masks == 1] = 2

        # Inverse initial transformations like normalization, polar transform, etc.
        if self.inverse_transform is not None:
            images, masks, preds = self.inverse_transform(images, masks, preds)

        # Calculate metrics
        if history is not None:
            update_metrics(masks, preds, history, self.labels)
            history['loss'].append(loss.item())

        return images, masks, preds

    def train_one_epoch(self, loader):
        self.model.train()
        mean_metrics = None
        history = defaultdict(list)
        loop = tqdm(loader, total=len(loader), leave=True, desc='Training')

        # Training loop
        for batch_idx, (images, masks) in enumerate(loop):
            # Run one iteration of forward and backward pass
            self.run_one_iteration(images, masks, backward=True, history=history)

            # Display average metrics in progress bar
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            if self.postfix_metrics:
                postfix = {k: mean_metrics[k] for k in self.postfix_metrics if k in mean_metrics}
            else:
                postfix = mean_metrics
            postfix['learning_rate'] = self.get_learning_rate()
            loop.set_postfix(**postfix)

        return mean_metrics

    def validate_one_epoch(self, loader):
        self.model.eval()
        history = defaultdict(list)
        mean_metrics = None
        loop = tqdm(loader, total=len(loader), leave=True, desc='Validation')

        # Validation loop
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(loop):
                # Perform only forward pass (no backpropagation)
                self.run_one_iteration(images, masks, backward=False, history=history)

                # Display average validation metrics in progress bar
                mean_metrics = {k: np.mean(v) for k, v in history.items()}
                if self.postfix_metrics:
                    postfix = {k: mean_metrics[k] for k in self.postfix_metrics if k in mean_metrics}
                else:
                    postfix = mean_metrics
                loop.set_postfix(**postfix)

        return mean_metrics


class BinaryLogger:

    def __init__(self, log_dir: str = '.', interval: int = 1, log_to_wandb: bool = False, show_plots: bool = False,
                 plot_type: str = 'all', class_labels: dict = None, num_examples: int = 4, part: str = 'validation',
                 binary_labels=None, threshold: float = 0.5):
        plot_type = plot_type.lower()
        assert plot_type in ('all', 'random', 'extreme', 'best', 'worst', 'OD', 'OC', 'none', '')
        assert binary_labels is not None, 'target class ids must be specified for binary segmentation'
        self.dir = log_dir
        self.interval = interval
        self.wandb = log_to_wandb
        self.show = show_plots
        self.plot_type = plot_type
        self.class_labels = class_labels
        self.num_examples = num_examples
        self.part = part
        self.binary_labels = binary_labels
        self.threshold = threshold

    def __call__(self, model, loader, optimizer, history, epoch, device, force: bool = False):
        if self.wandb:
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
            wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)
            wandb.log({f'max_{k}': np.max(v) for k, v in history.items() if 'loss' not in k}, step=epoch)

        if not force and (not self.interval or epoch % self.interval != 0):
            return

        optic = 'OC' if self.binary_labels.detach().cpu().numpy().tolist() == [2] else 'OD'

        model.eval()
        with torch.no_grad():
            images, masks = next(iter(loader))
            images = images.float().to(device)
            masks = masks.long().to(device)

            masks = torch.where(torch.isin(masks, self.binary_labels), 1, 0)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).squeeze(1).long()

            if optic == 'OC':
                preds[preds == 1] = 2
                masks[masks == 1] = 2

            images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            masks = masks.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            images = images[:self.num_examples]
            masks = masks[:self.num_examples]
            preds = preds[:self.num_examples]

        if self.wandb:  # Log interactive segmentation results to wandb
            for image, mask, pred in zip(images, masks, preds):
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
                wandb.log({f'Segmentation results for {optic} ({self.part})': seg_img}, step=epoch)
                break

        if not self.dir:
            return

        file = f'{self.dir}/epoch{epoch}.png'
        file_best_od = f'{self.dir}/epoch{epoch}_Best-OD.png'
        file_worst_od = f'{self.dir}/epoch{epoch}_Worst-OD.png'
        file_best_oc = f'{self.dir}/epoch{epoch}_Best-OC.png'
        file_worst_oc = f'{self.dir}/epoch{epoch}_Worst-OC.png'
        plot_types = ['image', 'mask', 'prediction', optic + ' overlap']

        if self.plot_type in ['all', 'random']:
            plot_results(images, masks, preds, save_path=file, show=self.show, types=plot_types)
            if self.wandb:
                wandb.log({f'Plotted results ({self.part})': wandb.Image(file)}, step=epoch)

        if self.plot_type in ['all', 'extreme', 'best', 'worst', 'OD'] and optic == 'OD':
            best, worst = get_best_and_worst_OD_examples(
                model, loader, self.num_examples, device=device, thresh=self.threshold)

            b0, b1, b2 = [e[0] for e in best], [e[1] for e in best], [e[2] for e in best]
            w0, w1, w2 = [e[0] for e in worst], [e[1] for e in worst], [e[2] for e in worst]

            if self.plot_type in ['all', 'extreme', 'best', 'OD']:
                plot_results(b0, b1, b2, save_path=file_best_od, show=self.show, types=plot_types)
                if self.wandb:
                    wandb.log({f'Best OD examples ({self.part})': wandb.Image(file_best_od)}, step=epoch)
            if self.plot_type in ['all', 'extreme', 'worst', 'OD']:
                plot_results(w0, w1, w2, save_path=file_worst_od, show=self.show, types=plot_types)
                if self.wandb:
                    wandb.log({f'Worst OD examples ({self.part})': wandb.Image(file_worst_od)}, step=epoch)

        if self.plot_type in ['all', 'extreme', 'best', 'worst', 'OC'] and optic == 'OC':
            best, worst = get_best_and_worst_OC_examples(
                model, loader, self.num_examples, device=device, thresh=self.threshold)

            b0, b1, b2 = [e[0] for e in best], [e[1] for e in best], [e[2] for e in best]
            w0, w1, w2 = [e[0] for e in worst], [e[1] for e in worst], [e[2] for e in worst]

            if self.plot_type in ['all', 'extreme', 'best', 'OC']:
                plot_results(b0, b1, b2, save_path=file_best_oc, show=self.show, types=plot_types)
                if self.wandb:
                    wandb.log({f'Best OC examples ({self.part})': wandb.Image(file_best_oc)}, step=epoch)
            if self.plot_type in ['all', 'extreme', 'worst', 'OC']:
                plot_results(w0, w1, w2, save_path=file_worst_oc, show=self.show, types=plot_types)
                if self.wandb:
                    wandb.log({f'Worst OC examples ({self.part})': wandb.Image(file_worst_oc)}, step=epoch)
