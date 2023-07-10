from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from utils.metrics import update_metrics, get_best_and_worst_OD_examples, get_best_and_worst_OC_examples
from utils.visualization import plot_results

__all__ = ['MulticlassTrainer', 'MulticlassTrainLogger']


class MulticlassTrainer:

    def __init__(self, model, criterion, optimizer, device, scaler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.od_label = [1, 2]
        self.oc_label = [2]
        self.labels = [self.od_label, self.oc_label]

    def train_one_epoch(self, loader):
        history = defaultdict(list)
        loop = tqdm(loader, total=len(loader), leave=True, desc='Training')
        mean_metrics = None

        # Set model to train mode
        self.model.train()
        # Iterate once over all the batches in the training data loader
        for batch_idx, (images, masks) in enumerate(loop):
            # Move data to device
            images = images.float().to(self.device)
            masks = masks.long().to(self.device)

            if self.scaler is None:
                # Forward pass
                outputs = self.model(images)  # Model returns logits
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

            # Convert logits to probabilities and then to label predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Calculate metrics
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

        # Set model to evaluation mode
        self.model.eval()
        # Disable gradient calculation
        with torch.no_grad():
            # Iterate once over all batches in the validation dataset
            for batch_idx, (images, masks) in enumerate(loop):
                images = images.float().to(self.device)
                masks = masks.long().to(self.device)

                # Forward pass only, no gradient calculation
                if self.scaler is None:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                else:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)

                # Convert logits to probabilities and predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                # Calculate metrics
                update_metrics(masks, preds, history, self.labels)
                history['loss'].append(loss.item())

                # Show summary of metrics in progress bar
                mean_metrics = {k: np.mean(v) for k, v in history.items()}
                loop.set_postfix(**mean_metrics)

        return mean_metrics


class MulticlassTrainLogger:

    def __init__(self, log_dir: str = '.', interval: int = 1, log_to_wandb: bool = False, show_plots: bool = False,
                 plot_type: str = 'all', class_labels: dict = None, num_examples: int = 4, part: str = 'validation'):
        self.dir = log_dir
        self.interval = interval
        self.wandb = log_to_wandb
        self.show = show_plots
        self.plot_type = plot_type
        self.class_labels = class_labels
        self.num_examples = num_examples
        self.part = part

    def __call__(self, model, loader, optimizer, history, epoch, device, force: bool = False):
        if self.wandb:  # Log performance metrics
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
            wandb.log({k: v[-1] for k, v in history.items()}, step=epoch)

        if not force and (not self.interval or epoch % self.interval != 0):
            return

        model.eval()
        with torch.no_grad():
            images, masks = next(iter(loader))
            images = images.float().to(device)
            masks = masks.long().to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            masks = masks.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            images = images[:self.num_examples]
            masks = masks[:self.num_examples]
            preds = preds[:self.num_examples]

        if self.wandb and self.class_labels is not None and self.plot_type not in ['none', '']:
            # Log interactive segmentation results to wandb
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
                wandb.log({f'Segmentation results ({self.part})': seg_img}, step=epoch)
                break

        file = f'{self.dir}/epoch{epoch}.png'
        file_best_od = f'{self.dir}/epoch{epoch}_Best-OD.png'
        file_worst_od = f'{self.dir}/epoch{epoch}_Worst-OD.png'
        file_best_oc = f'{self.dir}/epoch{epoch}_Best-OC.png'
        file_worst_oc = f'{self.dir}/epoch{epoch}_Worst-OC.png'
        plot_types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover']

        if self.plot_type in ['all', 'random']:
            plot_results(images, masks, preds, save_path=file, show=self.show, types=plot_types)
            if self.wandb:
                wandb.log({f'Plotted results ({self.part})': wandb.Image(file)}, step=epoch)

        if self.plot_type in ['all', 'extreme', 'best', 'worst', 'OD']:
            best, worst = get_best_and_worst_OD_examples(model, loader, self.num_examples, device=device)

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

        if self.plot_type in ['all', 'extreme', 'best', 'worst', 'OC']:
            best, worst = get_best_and_worst_OC_examples(model, loader, self.num_examples, device=device)

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
