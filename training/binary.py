from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from utils.metrics import update_metrics, get_best_and_worst_OD_examples, get_best_and_worst_OC_examples
from utils.visualization import plot_results

__all__ = ['BinaryTrainer', 'BinaryTrainLogger']


class BinaryTrainer:

    def __init__(self, model, criterion, optimizer, device, scaler=None, target_ids=None, threshold: float = 0.5,
                 inverse_transform=None, activation=None):
        assert target_ids is not None, 'target_ids must be specified for binary segmentation'
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.inverse_transform = inverse_transform
        self.activation = torch.sigmoid if activation is None else activation
        self.target_ids = target_ids
        self.threshold = threshold
        self.labels = [target_ids.detach().cpu().numpy().tolist()]

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def train_one_epoch(self, loader):
        history = defaultdict(list)
        loop = tqdm(loader, total=len(loader), leave=True, desc='Training')
        mean_metrics = None

        # Training loop
        self.model.train()
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.float().to(self.device)
            masks = masks.long().to(self.device)

            # Binarize mask by setting target ID labels to 1 and all other labels to 0 in mask
            masks = torch.where(torch.isin(masks, self.target_ids), 1, 0)

            if self.scaler is None:
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Convert logits to probabilities and apply threshold to get predictions
            probs = self.activation(outputs)
            preds = (probs > self.threshold).squeeze(1).long()

            # Shift predicted class labels for OC
            if self.labels == [[2]]:
                preds[preds == 1] = 2
                masks[masks == 1] = 2

            # Inverse initial transformations like normalization, polar transform, etc.
            if self.inverse_transform is not None:
                images, masks, preds = self.inverse_transform(images, masks, preds)

            # Calculate metrics
            update_metrics(masks, preds, history, self.labels)
            history['loss'].append(loss.item())

            # Display average metrics in progress bar
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics, learning_rate=self.get_learning_rate())

        return mean_metrics

    def validate_one_epoch(self, loader):
        history = defaultdict(list)
        loop = tqdm(loader, total=len(loader), leave=True, desc='Validation')
        mean_metrics = None

        # Validation loop
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(loop):
                # Load and prepare data
                images = images.float().to(self.device)
                masks = masks.long().to(self.device)
                masks = torch.where(torch.isin(masks, self.target_ids), 1, 0)

                # Forward pass
                if self.scaler is None:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                else:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)

                # Convert logits to probabilities
                probs = self.activation(outputs)
                preds = (probs > self.threshold).squeeze(1).long()

                # Correct predicted class labels for OC
                if self.labels == [[2]]:
                    preds[preds == 1] = 2
                    masks[masks == 1] = 2

                # Inverse initial data transformations on images, masks and predictions
                if self.inverse_transform is not None:
                    images, masks, preds = self.inverse_transform(images, masks, preds)

                # Calculate and update tracked metrics
                update_metrics(masks, preds, history, self.labels)
                history['loss'].append(loss.item())

                # Display average metrics in progress bar
                mean_metrics = {k: np.mean(v) for k, v in history.items()}
                loop.set_postfix(**mean_metrics)

        return mean_metrics

    def run_one_iteration(self, images, masks, backward: bool, history=None):
        images = images.float().to(self.device)
        masks = masks.long().to(self.device)

        # Binarize mask by setting target ID labels to 1 and all other labels to 0 in mask
        masks = torch.where(torch.isin(masks, self.target_ids), 1, 0)

        if self.scaler is None:
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        else:
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward pass
            if backward:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        # Convert logits to probabilities and apply threshold to get predictions
        probs = self.activation(outputs)
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


class BinaryTrainLogger:

    def __init__(self, log_dir: str = '.', interval: int = 1, log_to_wandb: bool = False, show_plots: bool = False,
                 plot_type: str = 'all', class_labels: dict = None, num_examples: int = 4, part: str = 'validation',
                 target_ids=None, threshold: float = 0.5):
        self.dir = log_dir
        self.interval = interval
        self.wandb = log_to_wandb
        self.show = show_plots
        self.plot_type = plot_type
        self.class_labels = class_labels
        self.num_examples = num_examples
        self.part = part
        self.target_ids = target_ids
        self.threshold = threshold

    def __call__(self, model, loader, optimizer, history, epoch, device, force: bool = False):
        if self.wandb:
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
            wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)

        if not force and (not self.interval or epoch % self.interval != 0):
            return

        assert self.target_ids is not None, 'target_ids must be specified for binary segmentation'
        optic = 'OC' if self.target_ids.detach().cpu().numpy().tolist() == [2] else 'OD'

        model.eval()
        with torch.no_grad():
            images, masks = next(iter(loader))
            images = images.float().to(device)
            masks = masks.long().to(device)

            masks = torch.where(torch.isin(masks, self.target_ids), 1, 0)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
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
        plot_types = ['image', 'mask', 'prediction', optic + ' cover']

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
