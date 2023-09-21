from collections import defaultdict
from IPython.display import clear_output
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from .binary import BinaryTrainer, BinaryLogger
from .multiclass import MulticlassTrainer, MulticlassLogger
from .multilabel import MultilabelTrainer, MultilabelLogger
from .cascade import CascadeTrainer, CascadeLogger
from .dual import DualTrainer, DualLogger
from utils.losses import *
from networks import *

__all__ = [
    'init_weights', 'update_history', 'save_checkpoint', 'load_checkpoint',
    'train', 'train_multiclass', 'train_multilabel', 'train_binary', 'train_cascade', 'train_dual',
    'get_network', 'get_optimizer', 'get_criterion', 'get_scheduler', 'get_scaler',
]

CLASS_LABELS = {
    0: 'Background',
    1: 'Optic Disc',
    2: 'Optic Cup',
}


# kwargs:
# early_stopping_patience, save_best_model, save_interval, log_interval, log_to_wandb, show_plots,
# clear_interval, checkpoint_dir, log_dir, plot_examples, inverse_transform, activation
def train_multiclass(model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                     scheduler=None, scaler=None, **kwargs):
    assert model.out_channels > 1, 'The model should have more than 1 output channel for multi-class training'
    return train(
        model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler,
        train_mode='multiclass', **kwargs,
    )


def train_multilabel(model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                     scheduler=None, scaler=None, threshold: float = 0.5, **kwargs):
    assert model.out_channels > 1, 'The model should have more than 1 output channel for multi-label training'
    return train(
        model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler,
        train_mode='multilabel', threshold=threshold, **kwargs,
    )


def train_binary(model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                 scheduler=None, scaler=None, target_ids: list[int] = None, threshold: float = 0.5, **kwargs):
    assert model.out_channels == 1, 'The model should have 1 output channel for binary training'
    return train(
        model=model, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler,
        train_mode='binary', target_ids=target_ids, threshold=threshold, **kwargs,
    )


def train_cascade(od_model, oc_model, criterion, optimizer, epochs, device, train_loader, val_loader=None,
                  scheduler=None, scaler=None, od_threshold: float = 0.5, oc_threshold: float = 0.5, **kwargs):
    assert od_model.out_channels == 1 and oc_model.out_channels == 1, \
        'The cascade models should have each 1 output channel for cascade training'
    return train(
        model=oc_model, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler, train_mode='cascade',
        od_threshold=od_threshold, oc_threshold=oc_threshold, base_cascade_model=od_model, **kwargs,
    )


def train_dual(model, od_criterion, oc_criterion, optimizer, epochs, device, train_loader, val_loader=None,
               scheduler=None, scaler=None, od_threshold: float = 0.5, oc_threshold: float = 0.5,
               od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0, **kwargs):
    assert model.out_channels == 1, 'The dual model should have 1 output channel per branch for dual training'
    return train(
        model=model, criterion=od_criterion, optimizer=optimizer, epochs=epochs, device=device,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, scaler=scaler, train_mode='dual',
        od_threshold=od_threshold, oc_threshold=oc_threshold, dual_branch_criterion=oc_criterion,
        od_loss_weight=od_loss_weight, oc_loss_weight=oc_loss_weight, **kwargs,
    )


def train(model, criterion, optimizer, epochs, device, train_loader, val_loader=None, scheduler=None, scaler=None,
          train_mode: str = 'multiclass', early_stopping_patience: int = 0, save_best_model: bool = True,
          save_interval: int = 0, log_interval: int = 0, log_to_wandb: bool = False, show_plots: bool = False,
          clear_interval: int = 5, checkpoint_dir: str = '.', log_dir: str = '.', plot_examples: str = 'all',
          target_ids: list[int] = None, threshold: float = 0.5, inverse_transform=None, activation=None,
          base_cascade_model=None, od_threshold: float = 0.5, oc_threshold: float = 0.5,
          dual_branch_criterion=None, od_loss_weight: float = 1.0, oc_loss_weight: float = 1.0):
    # model: model to train
    # criterion: loss function
    # optimizer: optimizer for gradient descent
    # epochs: number of epochs to train
    # device: 'cuda' or 'cpu'
    # train_loader: data loader with training set
    # val_loader: data loader with validation set
    # scheduler: learning rate scheduler (Optional)
    # scaler: scaler for mixed precision training (Optional)
    # train_mode: type of training ('multiclass', 'binary', 'cascade', 'dual')
    # early_stopping_patience: number of epochs to wait before training is stopped if the loss does not improve (0 or None to disable)
    # save_best_model: save the model with the best validation loss (True or False)
    # save_interval: save the model every few epochs (0 or None to disable)
    # log_interval: log tracked metrics and created plots every few epochs (0 or None to disable)
    # log_to_wandb: log progress to Weights & Biases (True or False)
    # show_plots: show examples from validation set (True or False)
    # clear_interval: clear text from cell output after every couple epoch (0 or None to disable)
    # checkpoint_dir: directory to save checkpoints (default: current directory, None to disable)
    # log_dir: directory to save logs (default: current directory, None to disable)
    # plot_examples: type of plots to create ('all', 'none', 'best', 'worst', 'extreme', 'OD', 'OC')
    # target_ids: defines which labels are considered as positives for binary segmentation (default: [1, 2])
    # threshold: threshold for predicted probabilities in binary training (default: 0.5)
    # inverse_transform: function to convert images, masks and predictions to original format (default: None)
    # activation: activation function for the last layer of a binary branch (default: None)
    # base_cascade_model: pre-trained model for optic disc segmentation for cascade architecture
    # od_threshold: decides whether a predicted optic disc probability is considered as a positive sample (default: 0.5)
    # oc_threshold: decides whether a predicted optic cup probability is considered as a positive sample (default: 0.5)
    # dual_branch_criterion: loss function for the second branch in dual branch training
    # od_loss_weight: weight for optic disc loss (default: 1.0)
    # oc_loss_weight: weight for optic cup loss (default: 1.0)
    # returns: history of training and validation metrics as a dictionary of lists

    assert train_mode in ('multiclass', 'multilabel', 'binary', 'cascade', 'dual'), \
        'Invalid training mode. Must be one of: multiclass, multilabel, binary, cascade, dual'

    history = defaultdict(list)
    best_loss = float('inf')
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    num_done_epochs = 0
    log_loader = val_loader if val_loader is not None else train_loader

    # Prepare model
    model_name = model.__class__.__name__
    model = model.to(device)
    if log_to_wandb:
        wandb.watch(model, criterion)

    # Initialize objects for selected training mode
    match train_mode:
        case 'multiclass':
            trainer = MulticlassTrainer(model, criterion, optimizer, device, scaler, inverse_transform)
            log = MulticlassLogger(log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS)
        case 'multilabel':
            trainer = MultilabelTrainer(
                model, criterion, optimizer, device, scaler, threshold, inverse_transform, activation,
            )
            log = MultilabelLogger(
                log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS, threshold=threshold,
            )
        case 'binary':
            if target_ids is None:
                target_ids = [1, 2]
            target_ids = torch.tensor(target_ids, device=device)

            trainer = BinaryTrainer(
                model, criterion, optimizer, device, scaler, target_ids, threshold, inverse_transform, activation,
            )
            log = BinaryLogger(
                log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS,
                target_ids=target_ids, threshold=threshold,
            )
        case 'cascade':
            # Prepare and freeze the pre-trained model
            base_cascade_model.eval()
            for param in base_cascade_model.parameters():
                param.requires_grad = False
            base_cascade_model = base_cascade_model.to(device)

            trainer = CascadeTrainer(
                base_cascade_model, model, criterion, optimizer, device, scaler,
                od_threshold, oc_threshold, inverse_transform, activation,
            )
            log = CascadeLogger(
                log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS,
                base_model=base_cascade_model, od_threshold=od_threshold, oc_threshold=oc_threshold,
            )
        case 'dual':
            trainer = DualTrainer(
                model, criterion, dual_branch_criterion, optimizer, device, scaler,
                od_threshold, oc_threshold, od_loss_weight, oc_loss_weight, inverse_transform, activation,
            )
            log = DualLogger(
                log_dir, log_interval, log_to_wandb, show_plots, plot_examples, CLASS_LABELS,
                od_threshold=od_threshold, oc_threshold=oc_threshold,
            )
        case _:
            raise ValueError(f'Invalid training mode: {train_mode}')

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
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, filename=f'{train_mode}-{model_name}-model-epoch{epoch}.pth', checkpoint_dir=checkpoint_dir)

        # Early stopping - stop training if the validation loss does not improve for a few epochs
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_metrics = {k: v[-1] for k, v in history.items()}
            best_epoch = epoch
            epochs_without_improvement = 0

            if save_best_model and checkpoint_dir:
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'history': history,
                }, filename=f'best-{train_mode}-{model_name}-model.pth', checkpoint_dir=checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement == early_stopping_patience:
                print(f'=> Early stopping: best validation loss at epoch {best_epoch} with metrics {best_metrics}')
                break

    # If last epoch was not logged, force log metrics before training ends
    if log_interval and num_done_epochs % log_interval != 0:
        log(model, log_loader, optimizer, history, num_done_epochs, device, force=True)

    return history


def initialize_weights(model, mode='fan_in', nonlinearity='relu'):
    for m in model.modules():
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
    print('Initialized weights of model:', model.__class__.__name__)


def init_weights(net, init_type='kaiming', gain=0.02):
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


def save_checkpoint(state, filename='model.pth', checkpoint_dir='.'):
    # create save directory if not exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # save model checkpoint
    filename = os.path.join(checkpoint_dir, filename)
    print(f'=> Saving checkpoint: {filename}')
    torch.save(state, filename)


def load_checkpoint(filename, model, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(f'Checkpoint file not found: {filename}')

    # load model weights
    print(f'=> Loading checkpoint: {filename}')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])


def get_network(name, in_channels, out_channels, layers, **kwargs):
    name = name.lower()

    if name == 'unet':
        return Unet(in_channels, out_channels, layers, **kwargs)
    if name == 'unet++':
        return UnetPlusPlus(in_channels, out_channels, layers, **kwargs)
    if name == 'unet3+':
        return Unet3Plus(in_channels, out_channels, layers, **kwargs)
    if name == 'resunet':
        return ResUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'runet':
        return RUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'r2unet':
        return R2Unet(in_channels, out_channels, layers, **kwargs)
    if name == 'attentionunet':
        return AttentionUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'doubleattentionunet':
        return DoubleAttentionUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'attentionsqueezeunet':
        return AttentionSqueezeUnet(in_channels, out_channels, **kwargs)
    if name == 'squeezeunet':
        return SqueezeUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'useunet':
        return USEnet(in_channels, out_channels, layers, **kwargs)
    if name == 'sarunet':
        return SARUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'inceptionunet':
        return InceptionUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'googleunet':
        return GoogleUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'r2attentionunet':
        return R2AttentionUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'r2unet++':
        return R2UnetPlusPlus(in_channels, out_channels, layers, **kwargs)
    if name == 'raunet++':
        return RAUnetPlusPlus(in_channels, out_channels, layers, **kwargs)
    if name == 'refunet3+cbam':
        return RefUnet3PlusCBAM(in_channels, out_channels, layers, **kwargs)

    if name == 'dualunet':
        return DualUnet(in_channels, out_channels, layers, **kwargs)
    # if name == 'dualunet++':
    #     return DualUnetPlusPlus(in_channels, out_channels, layers, **kwargs)
    # if name == 'dualunet3+':
    #     return DualUnet3Plus(in_channels, out_channels, layers, **kwargs)
    if name == 'dualresunet':
        return DualResUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'dualrunet':
        return DualRUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'dualr2unet':
        return DualR2Unet(in_channels, out_channels, layers, **kwargs)
    if name == 'dualattentionunet':
        return DualAttentionUnet(in_channels, out_channels, layers, **kwargs)
    # if name == 'dualdoubleattentionunet':
    #     return DualDoubleAttentionUnet(in_channels, out_channels, layers, **kwargs)
    # if name == 'dualattentionsqueezeunet':
    #     return DualAttentionSqueezeUnet(in_channels, out_channels, **kwargs)
    if name == 'dualsqueezeunet':
        return DualSqueezeUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'dualuseunet':
        return DualUSEnet(in_channels, out_channels, layers, **kwargs)
    if name == 'dualsarunet':
        return DualSARUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'dualinceptionunet':
        return DualInceptionUnet(in_channels, out_channels, layers, **kwargs)
    if name == 'dualgoogleunet':
        return DualGoogleUnet(in_channels, out_channels, layers, **kwargs)
    # if name == 'dualr2attentionunet':
    #     return DualR2AttentionUnet(in_channels, out_channels, layers, **kwargs)
    # if name == 'dualr2unet++':
    #     return DualR2UnetPlusPlus(in_channels, out_channels, layers, **kwargs)
    if name == 'dualraunet++':
        return DualRAUnetPlusPlus(in_channels, out_channels, layers, **kwargs)
    if name == 'dualrefunet3+cbam':
        return DualRefUnet3PlusCBAM(in_channels, out_channels, layers, **kwargs)


def get_optimizer(name, parameters, **kwargs):
    name = name.lower()

    if name == 'adam':
        return optim.Adam(parameters, **kwargs)
    if name == 'sgd':
        return optim.SGD(parameters, **kwargs)
    if name == 'rmsprop':
        return optim.RMSprop(parameters, **kwargs)
    raise ValueError(f'Invalid optimizer name: {name}')


def get_criterion(name, **kwargs):
    name = name.lower()

    if name == 'cross entropy':
        return nn.CrossEntropyLoss(**kwargs)
    if name == 'dice':
        return DiceLoss(**kwargs)
    if name == 'generalized dice':
        return GeneralizedDice(**kwargs)
    if name == 'iou':
        return IoULoss(**kwargs)
    if name == 'focal':
        return FocalLoss(**kwargs)
    if name == 'tversky':
        return TverskyLoss(**kwargs)
    if name == 'focal tversky':
        return FocalTverskyLoss(**kwargs)
    if name == 'hausdorff':
        return HausdorffLoss(**kwargs)
    if name == 'boundary':
        return BoundaryLoss(**kwargs)
    if name == 'ce':
        return CrossEntropyLoss(**kwargs)
    if name == 'ss':
        return SensitivitySpecificityLoss(**kwargs)
    if name == 'edge':
        return EdgeLoss(**kwargs)
    if name == 'combo':
        return ComboLoss(**kwargs)
    raise ValueError(f'Invalid loss function name: {name}')


def get_scheduler(name, optimizer, **kwargs):
    name = name.lower()

    if name == 'none':
        return None
    if name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    if name == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    if name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    if name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name == 'cosine warm restarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
    raise ValueError(f'Invalid scheduler name: {name}')


def get_scaler(name, **kwargs):
    name = name.lower()

    if name == 'none':
        return None
    if name == 'amp':
        return torch.cuda.amp.GradScaler(**kwargs)
    raise ValueError(f'Invalid scaler name: {name}')