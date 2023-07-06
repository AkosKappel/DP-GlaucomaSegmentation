from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from training.tools import CLASS_LABELS, update_history, save_checkpoint
from utils.metrics import update_metrics, get_best_and_worst_OD_examples, get_best_and_worst_OC_examples
from utils.visualization import plot_results

__all__ = ['train_multiclass']


def train_multiclass(model, criterion, optimizer, epochs, device, train_loader, val_loader=None, scheduler=None,
                     scaler=None, early_stopping_patience: int = 0, save_best_model: bool = True,
                     save_interval: int = 0, log_to_wandb: bool = False, show_plots: bool = False,
                     checkpoint_dir: str = '.', log_dir: str = '.', log_interval: int = 0, plot_examples: str = 'all'):
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
    # log_interval: log metrics and plots every few epochs (0 or None to disable)
    # plot_examples: type of plots to create ('all', 'none', 'best', 'worst', 'extreme', 'OD', 'OC')
    # returns: history of training and validation metrics as a dictionary of lists

    history = defaultdict(list)
    best_loss = np.inf
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    logger = MulticlassTrainLogger(log_dir, log_interval, log_to_wandb, show_plots, plot_examples)

    model = model.to(device)
    if log_to_wandb:
        wandb.watch(model, criterion)

    last_epoch = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')
        last_epoch = epoch

        # training
        train_metrics = train_one_multiclass_epoch(model, criterion, optimizer, device, train_loader, scaler)
        update_history(history, train_metrics, prefix='train')

        # skip validation part if data loader was not provided
        if val_loader is not None:
            # validation
            val_metrics = validate_one_multiclass_epoch(model, criterion, device, val_loader, scaler)
            update_history(history, val_metrics, prefix='val')

        val_loss = history['val_loss'][-1] if val_loader is not None else history['train_loss'][-1]

        # learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # log metrics locally and to wandb
        loader = val_loader if val_loader is not None else train_loader
        logger(model, loader, optimizer, history, epoch, device)

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

    # if last epoch was not logged, log metrics before ending training
    if last_epoch % log_interval != 0:
        loader = val_loader if val_loader is not None else train_loader
        logger(model, loader, optimizer, history, last_epoch, device, force=True)

    return history


def train_one_multiclass_epoch(model, criterion, optimizer, device, loader, scaler=None):
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


def validate_one_multiclass_epoch(model, criterion, device, loader, scaler=None):
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


class MulticlassTrainLogger:

    def __init__(self, log_dir: str = '.', interval: int = 1, log_to_wandb: bool = False, show_plots: bool = False,
                 plot_type: str = 'edge-case', num_examples: int = 4, part: str = 'validation'):
        self.dir = log_dir
        self.interval = interval
        self.wandb = log_to_wandb
        self.show = show_plots
        self.plot_type = plot_type
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

        if self.wandb:  # Log interactive segmentation results to wandb
            for image, mask, pred in zip(images, masks, preds):
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
