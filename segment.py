import albumentations as A
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from modules import *
from networks import *
from training import *
from ROI import CenterNet, preprocess_centernet_input, detect_roi

# Example usage:
# python segment.py ./ImagesForSegmentation --centernet ./models/roi/centernet.pth --roi-output-dir ./RoiResults
# python segment.py ./RoiResults -a dual -m ./models/polar/ref/dual.pth -o ./DualResults
# python segment.py ./RoiResults -a cascade -m ./models/polar/ref/cascade.pth --base-model ./models/polar/ref/binary.pth -o ./CascadeResults


def main():
    print('===========================================')
    print('Running inference for Glaucoma segmentation')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print('===========================================')

    parser = argparse.ArgumentParser(description='Glaucoma segmentation')
    parser.add_argument('path', type=str,
                        help='Path to image or directory of images to segment')
    parser.add_argument('-a', '--architecture', type=str,
                        help='Segmentation architecture', choices=('cascade', 'dual'))
    parser.add_argument('-m', '--model', type=str,
                        help='Path to segmentation model')
    parser.add_argument('-bm', '--base-model', type=str, default=None,
                        help='Path to base model for cascade segmentation')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for segmented images')
    parser.add_argument('-i', '--input-size', type=int, default=512,
                        help='Input image size for segmentation model')
    parser.add_argument('-od', '--optic-disc-threshold', type=float, default=0.2,
                        help='Optic cup threshold')
    parser.add_argument('-oc', '--optic-cup-threshold', type=float, default=0.3,
                        help='Optic cup threshold')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device used for inference', choices=('cuda', 'cpu'))
    parser.add_argument('-s', '--show', type=bool, default=False,
                        help='Show results in a plot')
    parser.add_argument('-c', '--centernet', type=str, default=None,
                        help='Path to CenterNet model if segmentation needs ROI detection step')
    parser.add_argument('--roi-input-size', type=int, default=512,
                        help='Input image size for CenterNet')
    parser.add_argument('--roi-output-dir', type=str, default=None,
                        help='Output directory for ROI images detected by CenterNet')

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('===========================================')

    if not os.path.exists(args.path):
        print(f'Path {args.path} does not exist')
        exit(1)

    files = load_files_from_dir(args.path)

    if args.centernet:
        files = apply_centernet(
            files, args.centernet,
            input_size=args.roi_input_size, output_size=args.input_size,
            device=args.device, show=args.show, output_dir=args.roi_output_dir,
        )

    if args.architecture and args.model:

        if not os.path.exists(args.model):
            print(f'Model {args.model} does not exist')
            exit(1)

        if args.architecture == 'cascade' and not os.path.exists(args.base_model):
            print(f'Base model {args.base_model} does not exist')
            exit(1)

        files = apply_segmentation(
            args.architecture, files, args.model, args.base_model,
            args.output_dir, args.optic_disc_threshold, args.optic_cup_threshold,
            args.input_size, args.device, args.show,
        )

    return files


def apply_segmentation(
        segmentation_mode: str, image_paths: list[str], model_path: str,
        base_model_path: str = None, output_dir: str = None,
        od_threshold: float = 0.2, oc_threshold: float = 0.3,
        input_size: int = 512, device: str = None, show: bool = False,
) -> list[np.ndarray]:
    assert segmentation_mode in ('cascade', 'dual')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load models
    ckpt = load_checkpoint(model_path, map_location=device)
    model = ckpt['model']
    model = model.to(device)
    model.eval()

    base_model = None
    if segmentation_mode == 'cascade':
        assert base_model_path, 'Please provide base model path'
        base_ckpt = load_checkpoint(base_model_path, map_location=device)
        base_model = base_ckpt['model']
        base_model = base_model.to(device)
        base_model.eval()

    # Preprocessing
    transform = A.Compose([
        A.Resize(height=input_size, width=input_size, interpolation=cv.INTER_AREA),
        A.Lambda(image=sharpen, p=1.0),
        A.Lambda(image=polar_transform, mask=polar_transform),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Create dataloader
    loader = load_dataset(
        image_paths,
        masks=None,
        transform=transform,
        batch_size=1,
        shuffle=False,
    )

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(image_paths[0]), f'{segmentation_mode.capitalize()}Results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    # Apply model inference
    for i, (image, _) in enumerate(tqdm(loader, desc=f'Segmenting images with {segmentation_mode} architecture')):
        prediction_mask, probability_maps, _ = predict(
            mode=segmentation_mode, model=model, images=image, masks=None,
            device=device, base_model=base_model, binary_labels=[1, 2], criterion=None,
            od_thresh=od_threshold, oc_thresh=oc_threshold, is_in_polar=True, tta=False,
            post_process_fn=postprocess, inter_process_fn=interprocess,
        )

        segmentation = prediction_mask.cpu().numpy().astype(np.uint8)
        segmentation = polar_to_cartesian(segmentation)[0]
        # od_probability_map = probability_maps[0, 0].cpu().numpy().astype(np.float32)
        # oc_probability_map = probability_maps[0, 1].cpu().numpy().astype(np.float32)

        # Calculate cCDR and show on original image
        original_image = cv.imread(image_paths[i], cv.IMREAD_COLOR)
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        contour_image = get_contour_image(
            original_image, segmentation, thickness=1,
            od_oc_colors=((0, 0, 0), (0, 0, 255)),  # RGB 0-255
        )
        vcdr = calculate_vCDR(segmentation)

        # Save results
        image_name = os.path.basename(image_paths[i])
        image_name_no_extension = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_dir, f'{image_name_no_extension}_vCDF={vcdr:.4f}.png')
        cv.imwrite(output_path, cv.cvtColor(contour_image, cv.COLOR_RGB2BGR))
        results.append(segmentation)

        # Display results
        if show:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
            ax[0].imshow(contour_image)
            ax[0].set_title(image_name)
            ax[1].imshow(segmentation)
            ax[1].set_title(f'vCDR = {vcdr}')
            plt.tight_layout()
            plt.show()

    return results


def apply_centernet(
        image_paths: list[str], model_path: str, output_dir: str = None,
        input_size: int = 512, output_size: int = 512,
        device: str = None, show: bool = False,
) -> list[str]:
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    state_dict = torch.load(model_path, map_location=device)
    model = CenterNet(n_classes=1, scale=4, base='resnet18', custom=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f'=> CenterNet model loaded from {model_path}')

    # Preprocessing
    transform = A.Compose([
        A.Resize(input_size, input_size, interpolation=cv.INTER_AREA),
        A.Normalize(mean=(0.9400, 0.6225, 0.3316), std=(0.1557, 0.1727, 0.1556)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    # Create output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(image_paths[0]), 'CenterNetResults')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for file in tqdm(image_paths, desc='Detecting ROIs with CenterNet'):
        if os.path.isdir(file):
            continue
        if not os.path.isfile(file):
            print(f'File not found: {file}')
            continue

        # Run ROI detection
        image_name = os.path.basename(file)
        image = preprocess_centernet_input(file)
        roi_image, _ = detect_roi(
            model=model, image_file=image, mask_file=None, device=device,
            transform=transform, input_size=input_size, roi_size=output_size,
            small_margin=32, large_margin=0, threshold=0.6,
        )

        # Save results
        result_path = os.path.join(output_dir, image_name)
        cv.imwrite(result_path, cv.cvtColor(roi_image, cv.COLOR_RGB2BGR))
        results.append(result_path)

        # Display results
        if show:
            plt.imshow(roi_image)
            plt.title(image_name)
            plt.show()

    print(f'{len(results)} ROI images saved to {output_dir}')
    return results


if __name__ == '__main__':
    main()
