import albumentations as A
import argparse
import cv2 as cv
import os
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2

from modules import *
from networks import *
from training import *

# Example usage:
# python train.py -a binary -m ref -o ./output --epochs 10
# python train.py -a cascade -m ref -o ./output --epochs 10 --base-model ./models/polar/ref/binary.pth
# python train.py -a dual -m ref -o ./output --epochs 10

TRAIN_IMAGES_DIRECTORIES = [
    './data/ORIGA/ROI/TrainImages',
    './data/DRISHTI/ROI/TrainImages',
]
TRAIN_MASKS_DIRECTORIES = [
    './data/ORIGA/ROI/TrainMasks',
    './data/DRISHTI/ROI/TrainMasks',
]
VAL_IMAGES_DIRECTORIES = [
    './data/ORIGA/ROI/TestImages',
    './data/DRISHTI/ROI/TestImages',
]
VAL_MASKS_DIRECTORIES = [
    './data/ORIGA/ROI/TestMasks',
    './data/DRISHTI/ROI/TestMasks',
]


def main():
    print('======================================')
    print('Running Glaucoma segmentation training')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print('======================================')

    parser = argparse.ArgumentParser(description='Glaucoma segmentation model training')
    parser.add_argument('-a', '--architecture', type=str, required=True,
                        help='Segmentation architecture', choices=('binary', 'cascade', 'dual'))
    parser.add_argument('-m', '--model-type', type=str, required=True,
                        help='Unet model variant', choices=('rau', 'ref', 'swin'))
    parser.add_argument('-bm', '--base-model', type=str, default=None,
                        help='Path to base model for cascade segmentation')
    parser.add_argument('-o', '--output-dir', type=str, default='./output',
                        help='Output directory for trained model')
    parser.add_argument('-bs', '--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('-i', '--input-size', type=int, default=256,
                        help='Input image size for segmentation model')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device used for training', choices=('cuda', 'cpu'))
    parser.add_argument('-p', '--polar', type=bool, default=True,
                        help='Use polar transformation')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--validate', type=bool, default=True,
                        help='Perform validation during training')

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('======================================')

    # Configurable parameters
    architecture = args.architecture
    model_type = args.model_type
    base_model_path = args.base_model
    output_dir = args.output_dir
    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs
    image_size = args.input_size
    polar = args.polar
    device = args.device
    dropout = args.dropout
    validate = args.validate

    # Hardcoded constants
    layers = [16, 32, 48, 64, 80]
    pin_memory = device == 'cuda'
    od_loss_weight = 1.0
    oc_loss_weight = 5.0
    in_channels = 3
    out_channels = 1
    num_workers = 0
    multi_scale_input = True
    deep_supervision = False
    early_stopping_patience = 10

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv.INTER_AREA),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomToneCurve(p=0.5),
        A.MultiplicativeNoise(p=0.5),
        A.Lambda(image=sharpen, p=1.0),
        A.Lambda(image=polar_transform, mask=polar_transform) if polar else A.Lambda(),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv.INTER_AREA),
        A.Lambda(image=sharpen, p=1.0),
        A.Lambda(image=polar_transform, mask=polar_transform) if polar else A.Lambda(),
        A.Normalize(),
        ToTensorV2(),
    ])

    train_loader = load_dataset(
        TRAIN_IMAGES_DIRECTORIES, TRAIN_MASKS_DIRECTORIES, train_transform,
        batch_size=batch_size, pin_memory=pin_memory,
        num_workers=num_workers, shuffle=True,
    )
    val_loader = None
    if validate:
        val_loader = load_dataset(
            VAL_IMAGES_DIRECTORIES, VAL_MASKS_DIRECTORIES, val_transform,
            batch_size=batch_size, pin_memory=pin_memory,
            num_workers=num_workers, shuffle=False,
        )

    model = None
    binary_model = None

    if architecture == 'dual' and 'dual-' not in model_type:
        model_type = 'dual-' + model_type

    if model_type == 'rau':
        model = RAUnetPlusPlus(
            in_channels=in_channels, out_channels=out_channels, features=layers,
            multi_scale_input=multi_scale_input, deep_supervision=deep_supervision, dropout=dropout,
        )

    if model_type == 'ref':
        model = RefUnet3PlusCBAM(
            in_channels=in_channels, out_channels=out_channels, features=layers,
            multi_scale_input=multi_scale_input, dropout=dropout,
        )

    if model_type == 'swin':
        model = SwinUnet(
            in_channels=in_channels, out_channels=out_channels, img_size=224, patch_size=4,
        )

    if model_type == 'dual-rau':
        model = DualRAUnetPlusPlus(
            in_channels=in_channels, out_channels=out_channels, features=layers,
            multi_scale_input=multi_scale_input, deep_supervision=deep_supervision, dropout=dropout,
        )

    if model_type == 'dual-ref':
        model = DualRefUnet3PlusCBAM(
            in_channels=in_channels, out_channels=out_channels, features=layers,
            multi_scale_input=multi_scale_input, dropout=dropout,
        )

    if model_type == 'dual-swin':
        model = DualSwinUnet(
            in_channels=in_channels, out_channels=out_channels, img_size=224, patch_size=4,
        )

    assert model is not None, f'Invalid network name: {model_type}'

    if architecture == 'dual' and 'dual-' in model_type:
        model_type = model_type.replace('dual-', '')

    model = model.to(device)
    init_model_weights(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ComboLoss(num_classes=1, class_weights=None)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
    scaler = None

    if architecture == 'cascade' and (not base_model_path or not os.path.isfile(base_model_path)):
        raise ValueError('Cascade architecture requires base model')

    if base_model_path:
        ckpt = load_checkpoint(base_model_path, map_location=device)
        binary_model = ckpt['model']

    hist = train(
        architecture, model, criterion, optimizer, epochs, device, train_loader, val_loader, scheduler, scaler,
        binary_labels=[1, 2],
        binary_model=binary_model, inter_processing=interprocess,
        od_loss_weight=od_loss_weight, oc_loss_weight=oc_loss_weight,
        save_interval=10, early_stopping_patience=early_stopping_patience,
        log_to_wandb=False, log_dir=None, log_interval=0, checkpoint_dir=output_dir,
        save_best_model=True, plot_examples='none', show_plots=False,
        inverse_transform=undo_polar_transform if polar else None,
        postfix_metrics=['accuracy_OD', 'accuracy_OC', 'dice_OD', 'dice_OC'],
    )

    print('=====================================')
    print('    Optic Disc results (training):')
    for k, v in hist.items():
        if 'OD' in k and 'train' in k:
            print(f'{k}: {v[-1]}')
    print('=====================================')
    print('    Optic Cup results (training):')
    for k, v in hist.items():
        if 'OC' in k and 'train' in k:
            print(f'{k}: {v[-1]}')
    print('=====================================')
    print('    Optic Disc results (validation):')
    for k, v in hist.items():
        if 'OD' in k and 'val' in k:
            print(f'{k}: {v[-1]}')
    print('=====================================')
    print('    Optic Cup results (validation):')
    for k, v in hist.items():
        if 'OC' in k and 'val' in k:
            print(f'{k}: {v[-1]}')
    print('=====================================')
    print('    Miscellaneous:')
    for k, v in hist.items():
        if 'OD' not in k and 'OC' not in k:
            print(f'{k}: {v[-1]}')

    metrics_to_plot = [
        'accuracy_OD', 'accuracy_OC',
        'balanced_accuracy_OD', 'balanced_accuracy_OC',
        'dice_OD', 'dice_OC',
        'iou_OD', 'iou_OC',
        'sensitivity_OD', 'sensitivity_OC',
        'specificity_OD', 'specificity_OC',
        'precision_OD', 'precision_OC',
        'loss',
    ]

    h = {k: v for k, v in hist.items() if any(k.endswith(m) for m in metrics_to_plot)}
    plot_history(h, figsize=(14, 12), save_path=f'{output_dir}/history-{architecture}-{model_type}.png')


if __name__ == '__main__':
    main()
