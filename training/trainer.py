import os
from collections import defaultdict

import torch
import torch.nn as nn
import wandb
from IPython.display import clear_output

from .binary import BinaryTrainer, BinaryLogger
from .multiclass import MulticlassTrainer, MulticlassLogger
from .multilabel import MultilabelTrainer, MultilabelLogger
from .cascade import CascadeTrainer, CascadeLogger
from .dual import DualTrainer, DualLogger

__all__ = [
    'train', 'train_multiclass', 'train_multilabel', 'train_binary', 'train_cascade', 'train_dual', 'fit',
    'init_model_weights', 'initialize_weights', 'update_history', 'save_checkpoint', 'load_checkpoint',
]

CLASS_LABELS = {
    0: 'Background',
    1: 'Optic Disc',
    2: 'Optic Cup',
}


class TrainingMode:
    MULTICLASS = 'multiclass'
    MULTILABEL = 'multilabel'
    BINARY = 'binary'
    CASCADE = 'cascade'
    DUAL = 'dual'


# Common kwargs:
#   early_stopping_patience, save_best_model, save_interval, log_interval, log_to_wandb, show_plots,
#   clear_interval, checkpoint_dir, log_dir, plot_examples, inverse_transform, activation
# Cascade only:
#   binary_model, inter_processing
# Dual only:
#   od_loss_weight, oc_loss_weight
def train(mode: str, model, criterion, optimizer, num_epochs: int, device, train_loader, val_loader=None,
          scheduler=None, scaler=None, binary_labels: list[int] = None, binary_model=None, inter_processing=None,
          od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0, **kwargs):
    if mode == TrainingMode.MULTICLASS:
        return train_multiclass(
            model, criterion, optimizer, num_epochs, device, train_loader, val_loader, scheduler, scaler, **kwargs,
        )

    if mode == TrainingMode.MULTILABEL:
        return train_multilabel(
            model, criterion, optimizer, num_epochs, device, train_loader, val_loader, scheduler, scaler, **kwargs,
        )

    if mode == TrainingMode.BINARY:
        assert binary_labels is not None, 'Binary class labels must be provided for binary segmentation'
        return train_binary(
            model, criterion, optimizer, num_epochs, device, train_loader, val_loader, scheduler, scaler,
            binary_labels=binary_labels, **kwargs,
        )

    if mode == TrainingMode.CASCADE:
        assert binary_model is not None, 'Base model for binary segmentation must be provided for cascade training'
        return train_cascade(
            binary_model, model, criterion, optimizer, num_epochs, device, train_loader, val_loader, scheduler, scaler,
            inter_processing=inter_processing, **kwargs,
        )

    if mode == TrainingMode.DUAL:
        return train_dual(
            model, criterion, criterion, optimizer, num_epochs, device, train_loader, val_loader, scheduler, scaler,
            od_loss_weight=od_loss_weight, oc_loss_weight=oc_loss_weight, **kwargs,
        )


def train_multiclass(model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                     scheduler=None, scaler=None, **kwargs):
    assert model.out_channels > 1, 'The model should have more than 1 output channel for multi-class training'
    return fit(
        model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler,
        train_mode=TrainingMode.MULTICLASS, **kwargs,
    )


def train_multilabel(model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                     scheduler=None, scaler=None, threshold: float = 0.5, **kwargs):
    assert model.out_channels > 1, 'The model should have more than 1 output channel for multi-label training'
    return fit(
        model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler,
        train_mode=TrainingMode.MULTILABEL, threshold=threshold, **kwargs,
    )


def train_binary(model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                 scheduler=None, scaler=None, binary_labels: list[int] = None, threshold: float = 0.5, **kwargs):
    assert model.out_channels == 1, 'The model should have 1 output channel for binary training'
    return fit(
        model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler,
        train_mode=TrainingMode.BINARY, binary_labels=binary_labels, threshold=threshold, **kwargs,
    )


def train_cascade(base_model, model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                  scheduler=None, scaler=None, od_threshold: float = 0.5, oc_threshold: float = 0.5,
                  inter_processing=None, **kwargs):
    assert model.out_channels == 1 and base_model.out_channels == 1, \
        'The cascade models should have each 1 output channel for cascade training'
    return fit(
        model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler,
        train_mode=TrainingMode.CASCADE, od_threshold=od_threshold, oc_threshold=oc_threshold,
        base_cascade_model=base_model, inter_processing=inter_processing, **kwargs,
    )


def train_dual(model, od_criterion, oc_criterion, optimizer, epochs, device, train_loader, val_loader=None,
               scheduler=None, scaler=None, od_threshold: float = 0.5, oc_threshold: float = 0.5,
               od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0, **kwargs):
    assert model.out_channels == 1, \
        'The dual decoder model should have 1 output channel per branch for dual training'
    return fit(
        model=model, criterion=od_criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler,
        train_mode=TrainingMode.DUAL, od_threshold=od_threshold, oc_threshold=oc_threshold,
        dual_branch_criterion=oc_criterion, od_loss_weight=od_loss_weight, oc_loss_weight=oc_loss_weight, **kwargs,
    )


def fit(model, criterion, optimizer, epochs, device, train_loader, val_loader=None, scheduler=None, scaler=None,
        train_mode: str = TrainingMode.MULTICLASS, early_stopping_patience: int = 0, save_best_model: bool = True,
        save_interval: int = 0, log_interval: int = 0, log_to_wandb: bool = False, show_plots: bool = False,
        clear_interval: int = 5, checkpoint_dir: str = '.', log_dir: str = '.', plot_examples: str = 'all',
        binary_labels: list[int] = None, threshold: float = 0.5, inverse_transform=None, activation=None,
        base_cascade_model=None, inter_processing=None, od_threshold: float = 0.5, oc_threshold: float = 0.5,
        dual_branch_criterion=None, od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0):
    """
    Train a model with a given criterion and optimizer for a specified number of epochs in a selected training mode.

    Args:
    - model: model to train
    - criterion: loss function
    - optimizer: optimizer for gradient descent
    - epochs: number of epochs to train
    - device: 'cuda' or 'cpu'
    - train_loader: data loader with training set
    - val_loader: data loader with validation set (Optional)
    - scheduler: learning rate scheduler (Optional)
    - scaler: scaler for mixed precision training (Optional)
    - train_mode: type of training ('multiclass', 'multilabel', 'binary', 'cascade', 'dual')
    - early_stopping_patience: number of epochs to wait before training is stopped if the loss does not improve (0 or None to disable)
    - save_best_model: save the model with the best validation loss (True or False)
    - save_interval: save the model every few epochs (0 or None to disable)
    - log_interval: log tracked metrics and created plots every few epochs (0 or None to disable)
    - log_to_wandb: log progress to Weights & Biases (True or False)
    - show_plots: show examples from validation set (True or False)
    - clear_interval: clear text from cell output after every couple epoch (0 or None to disable)
    - checkpoint_dir: directory to save checkpoints (default: current directory, None to disable)
    - log_dir: directory to save logs (default: current directory, None to disable)
    - plot_examples: type of plots to create ('all', 'none', 'best', 'worst', 'extreme', 'OD', 'OC')
    - target_ids: defines which labels are considered as positives for binary segmentation (default: [1, 2])
    - threshold: threshold for predicted probabilities in binary training (default: 0.5)
    - inverse_transform: function to convert images, masks and predictions to original format (default: None)
    - activation: activation function for the last layer of a binary branch (default: None for sigmoid)
    - base_cascade_model: pre-trained model for optic disc segmentation for cascade architecture
    - inter_processing: post-processing function for the optic disc segmentation in cascade architecture (Optional)
    - od_threshold: decides whether a predicted optic disc probability is considered as a positive sample (default: 0.5)
    - oc_threshold: decides whether a predicted optic cup probability is considered as a positive sample (default: 0.5)
    - dual_branch_criterion: loss function for the second branch in dual branch training (Optional)
    - od_loss_weight: weight for optic disc loss (default: 1.0)
    - oc_loss_weight: weight for optic cup loss (default: 1.0)

    Returns:
    - history: history of training and validation metrics as a dictionary of lists
               e.g. {'train_loss': [0.1, 0.05, ...], ..., 'val_dice': [0.9, 0.92, ...]}
    """
    # Prepare model
    model_name = model.__class__.__name__
    model = model.to(device)
    if log_to_wandb:
        wandb.watch(model, criterion)

    # Initialize objects for selected training mode
    train_mode = train_mode.lower()
    if train_mode == TrainingMode.MULTICLASS:
        trainer = MulticlassTrainer(model, criterion, optimizer, device, scaler, inverse_transform)
        log = MulticlassLogger(log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS)
    elif train_mode == TrainingMode.MULTILABEL:
        trainer = MultilabelTrainer(
            model, criterion, optimizer, device, scaler, threshold, inverse_transform, activation,
        )
        log = MultilabelLogger(
            log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS, threshold=threshold,
        )
    elif train_mode == TrainingMode.BINARY:
        if binary_labels is None:
            binary_labels = [1, 2]
        binary_labels = torch.tensor(binary_labels, device=device)

        trainer = BinaryTrainer(
            model, criterion, optimizer, device, scaler, binary_labels, threshold, inverse_transform, activation,
        )
        log = BinaryLogger(
            log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS,
            binary_labels=binary_labels, threshold=threshold,
        )
    elif train_mode == TrainingMode.CASCADE:
        # Prepare and freeze the pre-trained model
        base_cascade_model.eval()
        for param in base_cascade_model.parameters():
            param.requires_grad = False
        base_cascade_model = base_cascade_model.to(device)

        trainer = CascadeTrainer(
            base_cascade_model, model, criterion, optimizer, device, scaler,
            od_threshold, oc_threshold, inverse_transform, activation, inter_processing,
        )
        log = CascadeLogger(
            log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS,
            base_model=base_cascade_model, od_threshold=od_threshold, oc_threshold=oc_threshold,
        )
    elif train_mode == TrainingMode.DUAL:
        trainer = DualTrainer(
            model, criterion, dual_branch_criterion, optimizer, device, scaler,
            od_threshold, oc_threshold, od_loss_weight, oc_loss_weight, inverse_transform, activation,
        )
        log = DualLogger(
            log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS,
            od_threshold=od_threshold, oc_threshold=oc_threshold,
        )
    else:
        raise ValueError(f'Invalid training mode: {train_mode}')

    # Initialize tracking variables
    history = defaultdict(list)
    best_loss = float('inf')
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    num_done_epochs = 0
    log_loader = val_loader if val_loader is not None else train_loader

    # Run training & validation for N epochs
    for epoch in range(1, epochs + 1):
        # Empty notebook cell output after every few epochs
        if clear_interval and epoch % clear_interval == 0:
            clear_output(wait=True)

        # Start a new epoch
        print(f'Epoch {epoch}:')

        # Training
        train_metrics = trainer.train_one_epoch(train_loader)
        update_history(history, train_metrics, prefix='train')

        # Validation
        if val_loader is not None:
            val_metrics = trainer.validate_one_epoch(val_loader)
            update_history(history, val_metrics, prefix='val')

        num_done_epochs += 1
        epoch_loss = history['val_loss'][-1] if 'val_loss' in history else history['train_loss'][-1]

        # Learning rate scheduler - lower LR if needed for better convergence
        if scheduler is not None:
            scheduler.step(epoch_loss)

        # Logging - log metrics and plots locally and to Weights & Biases
        log(model, log_loader, optimizer, history, epoch, device)

        # Checkpoints - save model every few epochs
        if save_interval and epoch % save_interval == 0 and checkpoint_dir:
            save_checkpoint({
                'epoch': epoch,
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer,
                'history': history,
            }, filepath=f'{checkpoint_dir}/{train_mode}-{model_name}-model-epoch{epoch}.pth')

        # Early stopping - stop training if the validation loss does not improve for a few epochs
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_metrics = {k: v[-1] for k, v in history.items()}
            best_epoch = epoch
            epochs_without_improvement = 0

            if save_best_model and checkpoint_dir:
                save_checkpoint({
                    'epoch': epoch,
                    'model': model,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer,
                    'history': history,
                }, filepath=f'{checkpoint_dir}/best-{train_mode}-{model_name}-model.pth')
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement == early_stopping_patience:
                print(f'=> Early stopping: best validation loss at epoch {best_epoch} with metrics {best_metrics}')
                break

    # If last epoch was not logged, force log metrics before training ends
    if log_interval and num_done_epochs % log_interval != 0:
        log(model, log_loader, optimizer, history, num_done_epochs, device, force=True)

    return history


def initialize_weights(net, mode='fan_in', nonlinearity='relu'):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Use Kaiming initialization for ReLU activation function
            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
            # Use zero bias
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize weight to 1 and bias to 0
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    print(f'Initialized {net.__class__.__name__} model weights with mode={mode} and nonlinearity={nonlinearity}')


def init_model_weights(net, init_type: str = 'kaiming', gain: float = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    print(f'Initialized {net.__class__.__name__} network parameters with {init_type} method.')


def update_history(history, metrics, prefix=''):
    for k, v in metrics.items():
        history[f'{prefix}_{k}'].append(v)


def save_checkpoint(state, filepath: str = 'model.pth'):
    # create save directory if not exists
    checkpoint_dir = os.path.dirname(filepath)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # save checkpoint to file
    print(f'=> Saving checkpoint: {filepath}')
    torch.save(state, filepath)


def load_checkpoint(filepath: str, **kwargs):
    # check if file exists
    assert os.path.isfile(filepath), f'Invalid checkpoint file: {filepath}'

    # load checkpoint from file
    print(f'=> Loading checkpoint: {filepath}')
    checkpoint = torch.load(filepath, **kwargs)
    return checkpoint
