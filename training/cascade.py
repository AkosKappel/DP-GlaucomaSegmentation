from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from utils.metrics import update_metrics, get_best_and_worst_OD_examples, get_best_and_worst_OC_examples
from utils.visualization import plot_results

__all__ = ['CascadeTrainer', 'CascadeTrainLogger']


class CascadeTrainer:

    def __init__(self, od_model, oc_model, criterion, optimizer, device, scaler=None,
                 od_threshold: float = 0.5, oc_threshold: float = 0.5, inverse_transform=None, activation=None):
        self.od_model = od_model
        self.oc_model = oc_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.inverse_transform = inverse_transform
        self.activation = torch.sigmoid if activation is None else activation
        self.od_threshold = od_threshold
        self.oc_threshold = oc_threshold
        self.od_label = [1, 2]
        self.oc_label = [2]
        self.labels = [self.od_label, self.oc_label]

    def train_one_epoch(self, loader):
        history = defaultdict(list)
        loop = tqdm(loader, total=len(loader), leave=True, desc='Training')
        mean_metrics = None

        # Training loop
        self.od_model.eval()
        self.oc_model.train()
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.float().to(self.device)
            masks = masks.long().to(self.device)

            # Apply first model to get optic disc masks which get passed to the second model
            with torch.no_grad():
                od_outputs = self.od_model(images)
                od_probs = self.activation(od_outputs)
                od_preds = (od_probs > self.od_threshold).long()

            # Crop images to optic disc boundaries
            cropped_images = images * od_preds

            # Create optic cup only masks
            oc_masks = (masks == 2).long()

            if self.scaler is None:
                # Forward pass
                oc_outputs = self.oc_model(cropped_images)
                loss = self.criterion(oc_outputs, oc_masks)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                # Forward pass
                with torch.cuda.amp.autocast():
                    oc_outputs = self.oc_model(cropped_images)
                    loss = self.criterion(oc_outputs, oc_masks)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Convert logits to probabilities
            oc_probs = self.activation(oc_outputs)
            oc_preds = (oc_probs > self.oc_threshold).squeeze(1).long()

            od_preds = od_preds.squeeze(1)

            # Join OD and OC predictions into a single tensor
            preds = torch.zeros_like(oc_preds)
            preds[od_preds == 1] = 1
            preds[oc_preds == 1] = 2

            # Invert initial transformations like normalization, polar transform, etc.
            if self.inverse_transform is not None:
                images, masks, preds = self.inverse_transform(images, masks, preds)

            # Calculate metrics for the batch
            update_metrics(masks, preds, history, self.labels)
            history['loss'].append(loss.item())

            # Display average metrics in progress bar
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics, learning_rate=self.optimizer.param_groups[0]['lr'])

        return mean_metrics

    def validate_one_epoch(self, loader):
        history = defaultdict(list)
        loop = tqdm(loader, total=len(loader), leave=True, desc='Validation')
        mean_metrics = None

        # Validation loop
        self.od_model.eval()
        self.oc_model.eval()
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(loop):
                # Prepare data
                images = images.float().to(self.device)
                masks = masks.long().to(self.device)

                # Create optic disc masks
                od_outputs = self.od_model(images)
                od_probs = self.activation(od_outputs)
                od_preds = (od_probs > self.od_threshold).long()

                # Apply masks to images
                cropped_images = images * od_preds

                # Create optic cup only masks
                oc_masks = (masks == 2).long()

                # Forward pass (no backprop)
                if self.scaler is None:
                    oc_outputs = self.oc_model(cropped_images)
                    loss = self.criterion(oc_outputs, oc_masks)
                else:
                    with torch.cuda.amp.autocast():
                        oc_outputs = self.oc_model(cropped_images)
                        loss = self.criterion(oc_outputs, oc_masks)

                # Convert logits to predictions
                oc_probs = self.activation(oc_outputs)
                oc_preds = (oc_probs > self.oc_threshold).squeeze(1).long()

                od_preds = od_preds.squeeze(1)

                # Join OD and OC predictions
                preds = torch.zeros_like(oc_preds)
                preds[od_preds == 1] = 1
                preds[oc_preds == 1] = 2

                # Invert initial transformations like normalization, polar transform, etc.
                if self.inverse_transform is not None:
                    images, masks, preds = self.inverse_transform(images, masks, preds)

                # calculate metrics
                update_metrics(masks, preds, history, self.labels)
                history['loss'].append(loss.item())

                # show summary of metrics in progress bar
                mean_metrics = {k: np.mean(v) for k, v in history.items()}
                loop.set_postfix(**mean_metrics)

        return mean_metrics


class CascadeTrainLogger:

    def __init__(self, log_dir: str = '.', interval: int = 1, log_to_wandb: bool = False, show_plots: bool = False,
                 plot_type: str = 'all', class_labels: dict = None, num_examples: int = 4, part: str = 'validation',
                 base_model=None, od_threshold: float = 0.5, oc_threshold: float = 0.5):
        self.dir = log_dir
        self.interval = interval
        self.wandb = log_to_wandb
        self.show = show_plots
        self.plot_type = plot_type
        self.class_labels = class_labels
        self.num_examples = num_examples
        self.part = part
        self.base_model = base_model
        self.od_threshold = od_threshold
        self.oc_threshold = oc_threshold

    def __call__(self, model, loader, optimizer, history, epoch, device, force: bool = False):
        if self.wandb:
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
            wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)

        if not force and (not self.interval or epoch % self.interval != 0):
            return

        self.base_model.eval()
        model.eval()
        with torch.no_grad():
            images, masks = next(iter(loader))
            images = images.float().to(device)
            masks = masks.long().to(device)

            # Create optic disc masks
            od_outputs = self.base_model(images)
            od_probs = torch.sigmoid(od_outputs)
            od_preds = (od_probs > self.od_threshold).long()

            # Apply masks to images
            cropped_images = images * od_preds

            oc_masks = (masks == 2).long()

            oc_outputs = model(cropped_images)
            oc_probs = torch.sigmoid(oc_outputs)
            oc_preds = (oc_probs > self.oc_threshold).squeeze(1).long()

            oc_masks[oc_masks == 1] = 2
            oc_preds[oc_preds == 1] = 2

            images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            oc_masks = oc_masks.detach().cpu().numpy()
            oc_preds = oc_preds.detach().cpu().numpy()

            images = images[:self.num_examples]
            oc_masks = oc_masks[:self.num_examples]
            oc_preds = oc_preds[:self.num_examples]

        if self.wandb:  # Log interactive segmentation results
            for image, mask, pred in zip(images, oc_masks, oc_preds):
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

        file = f'{self.dir}/epoch{epoch}.png'
        file_best_od = f'{self.dir}/epoch{epoch}_Best-OD.png'
        file_worst_od = f'{self.dir}/epoch{epoch}_Worst-OD.png'
        file_best_oc = f'{self.dir}/epoch{epoch}_Best-OC.png'
        file_worst_oc = f'{self.dir}/epoch{epoch}_Worst-OC.png'
        plot_types_od = ['image', 'mask', 'prediction', 'OD cover']
        plot_types_oc = ['image', 'mask', 'prediction', 'OC cover']

        if self.plot_type in ['all', 'random']:
            plot_results(images, oc_masks, oc_preds, save_path=file, show=self.show, types=plot_types_od)
            if self.wandb:
                wandb.log({f'Plotted results ({self.part})': wandb.Image(file)}, step=epoch)

        if self.plot_type in ['all', 'extreme', 'best', 'worst', 'OD']:
            best, worst = get_best_and_worst_OD_examples(
                self.base_model, loader, self.num_examples, device=device, thresh=self.od_threshold)

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
                model, loader, self.num_examples, device=device, thresh=self.oc_threshold, first_model=self.base_model)

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
