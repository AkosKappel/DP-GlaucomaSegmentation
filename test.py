import albumentations as A
import argparse
import cv2 as cv
import os
import torch
from albumentations.pytorch import ToTensorV2

from modules import *
from networks import *
from training import *

# Example usage:
# python test.py -a binary -m ./models/polar/ref/binary.pth
# python test.py -a cascade -m ./models/polar/ref/cascade.pth --base-model ./models/polar/ref/binary.pth
# python test.py -a dual -m ./models/polar/ref/dual.pth


def main():
    print('=====================================')
    print('Running Glaucoma segmentation testing')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print('=====================================')

    parser = argparse.ArgumentParser(description='Glaucoma segmentation model testing')
    parser.add_argument('-a', '--architecture', type=str, required=True,
                        help='Segmentation architecture', choices=('binary', 'cascade', 'dual'))
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to segmentation model')
    parser.add_argument('-bm', '--base-model', type=str, default=None,
                        help='Path to base model for cascade segmentation')
    parser.add_argument('-i', '--input-size', type=int, default=256,
                        help='Input image size for segmentation model')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device used for training', choices=('cuda', 'cpu'))
    parser.add_argument('-p', '--polar', type=bool, default=True,
                        help='Use polar transformation')

    test_images_dir = [
        './data/ORIGA/ROI/TestImages',
        './data/DRISHTI/ROI/TestImages',
    ]
    test_masks_dir = [
        './data/ORIGA/ROI/TestMasks',
        './data/DRISHTI/ROI/TestMasks',
    ]

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('=====================================')

    # Configurable parameters
    architecture = args.architecture
    model_path = args.model
    base_model_path = args.base_model
    image_size = args.input_size
    polar = args.polar
    device = args.device

    # Hardcoded constants
    pin_memory = device == 'cuda'
    num_workers = 0

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    if not model_path or not os.path.isfile(model_path):
        raise ValueError('Model path not found')

    ckpt = load_checkpoint(model_path, map_location=device)
    model = ckpt['model']
    model = model.to(device)
    model.eval()

    binary_model = None
    if architecture == 'cascade':
        if not base_model_path or not os.path.isfile(base_model_path):
            raise ValueError('Cascade architecture requires base model')

        # Load base model
        ckpt = load_checkpoint(base_model_path, map_location=device)
        binary_model = ckpt['model']
        binary_model = binary_model.to(device)
        binary_model.eval()

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    test_transform = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv.INTER_AREA),
        A.Lambda(image=sharpen, p=1.0),
        A.Lambda(image=polar_transform, mask=polar_transform) if polar else A.Lambda(),
        A.Normalize(),
        ToTensorV2(),
    ])

    test_loader = load_dataset(
        test_images_dir, test_masks_dir, test_transform, batch_size=1,
        pin_memory=pin_memory, num_workers=num_workers, shuffle=True,
    )

    criterion = ComboLoss(num_classes=1, class_weights=None)

    results = evaluate(
        architecture, model, test_loader, device, criterion,
        binary_labels=[1, 2], base_model=binary_model,
        inverse_transform=undo_polar_transform if polar else None,
        inter_process_fn=interprocess, post_process_fn=postprocess, tta=False,
        postfix_metrics=['accuracy_OD', 'accuracy_OC', 'dice_OD', 'dice_OC'],
    )

    print('=====================================')
    print('    Optic Disc results:')
    for k, v in results.items():
        if 'OD' in k:
            print(f'{k}: {v}')
    print('=====================================')
    print('    Optic Cup results:')
    for k, v in results.items():
        if 'OC' in k:
            print(f'{k}: {v}')
    print('=====================================')
    print('    Miscellaneous:')
    for k, v in results.items():
        if 'OD' not in k and 'OC' not in k:
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
