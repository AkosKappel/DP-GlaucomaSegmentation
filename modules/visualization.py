import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchviz
from torchview import draw_graph
from matplotlib.patches import Patch

from modules.inference import predict

__all__ = [
    'get_input_images', 'get_input_image', 'get_ground_truth_masks', 'get_ground_truth_mask',
    'get_prediction_masks', 'get_prediction_mask', 'get_overlay_images', 'get_overlay_image',
    'get_cover_images', 'get_cover_image', 'get_contour_images', 'get_contour_image',
    'plot_history', 'show_model_graph', 'show_model_view',
    'plot_image_grid', 'plot_results', 'plot_results_from_loader',
]

# Label colors
BG_COLOR = np.array((0.27, 0.01, 0.33), dtype=np.float32)
OD_COLOR = np.array((0.13, 0.56, 0.55), dtype=np.float32)
OC_COLOR = np.array((0.99, 0.91, 0.14), dtype=np.float32)

# Overlay colors
TP_COLOR = np.array((0, 1, 0), dtype=np.uint8) * 255  # green
TN_COLOR = np.array((0, 0, 0), dtype=np.uint8) * 255  # black
FP_COLOR = np.array((1, 0, 0), dtype=np.uint8) * 255  # red
FN_COLOR = np.array((1, 1, 0), dtype=np.uint8) * 255  # yellow

# Cover colors
OD_COVER_COLOR = np.array((0, 0, 1))  # blue
OC_COVER_COLOR = np.array((0, 1, 1))  # cyan

# Contour colors
TRUE_CONTOUR_COLOR = (0, 0, 255)  # blue
PRED_CONTOUR_COLOR = (0, 0, 0)  # black


def get_input_images(imgs: list[np.ndarray]):
    return [get_input_image(img) for img in imgs]


def get_input_image(img: np.ndarray, normalize: bool = True, uint8: bool = False):
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    if normalize:
        img -= img.min()
        img /= img.max()
    if uint8:
        img = (img * 255).astype(np.uint8)
    return img


def get_ground_truth_masks(masks: list[np.ndarray]):
    return [get_ground_truth_mask(mask) for mask in masks]


def get_ground_truth_mask(mask: np.ndarray):
    gt = np.zeros((*mask.shape, 3), dtype=np.float32)
    gt[mask == 0] = BG_COLOR
    gt[mask == 1] = OD_COLOR
    gt[mask == 2] = OC_COLOR
    return gt


def get_prediction_masks(preds: list[np.ndarray]):
    return [get_prediction_mask(pred) for pred in preds]


def get_prediction_mask(pred: np.ndarray):
    pr = np.zeros((*pred.shape, 3), dtype=np.float32)
    pr[pred == 0] = BG_COLOR
    pr[pred == 1] = OD_COLOR
    pr[pred == 2] = OC_COLOR
    return pr


def get_overlay_images(masks: list[np.ndarray], preds: list[np.ndarray], **kwargs):
    return [get_overlay_image(mask, pred, **kwargs) for mask, pred in zip(masks, preds)]


def get_overlay_image(mask: np.ndarray, pred: np.ndarray, class_ids: list[int] = None):
    """
    Visualize the correctness of the segmentation, by overlaying the mask and the prediction and
    color coding the results with TP, TN, FP, FN colors. The colors are defined as follows:
    True Positive: green
    True Negative: black
    False Positive: red
    False Negative: yellow
    """
    cover_img = np.zeros((*mask.shape, 3), dtype=np.uint8)

    if class_ids is None:
        # treat image as binary (0 - background, non-zero - foreground)
        cover_img[(mask != 0) & (pred != 0)] = TP_COLOR
        cover_img[(mask == 0) & (pred == 0)] = TN_COLOR
        cover_img[(mask == 0) & (pred != 0)] = FP_COLOR
        cover_img[(mask != 0) & (pred == 0)] = FN_COLOR
    else:
        # True positive (TP): both mask and pred have one of the class_ids at the same pixel
        tp_mask = np.isin(mask, class_ids) & np.isin(pred, class_ids)
        cover_img[tp_mask] = TP_COLOR

        # True negative (TN): neither mask nor pred have one of the class_ids at the same position
        tn_mask = np.logical_not(np.isin(mask, class_ids)) & np.logical_not(np.isin(pred, class_ids))
        cover_img[tn_mask] = TN_COLOR

        # False positive (FP): pred has one of the class_ids, but mask does not
        fp_mask = np.logical_not(np.isin(mask, class_ids)) & np.isin(pred, class_ids)
        cover_img[fp_mask] = FP_COLOR

        # False negative (FN): pred does not have one of the class_ids, but mask does
        fn_mask = np.isin(mask, class_ids) & np.logical_not(np.isin(pred, class_ids))
        cover_img[fn_mask] = FN_COLOR

    return cover_img


def get_cover_images(imgs: list[np.ndarray], masks: list[np.ndarray], **kwargs):
    return [get_cover_image(img, mask, **kwargs) for img, mask in zip(imgs, masks)]


def get_cover_image(img: np.ndarray, mask: np.ndarray, alpha=0.3):
    """Draw the mask over the input image with a specified opacity."""
    img = get_input_image(img)

    overlay_img = np.zeros_like(img)
    overlay_img[mask == 1] = OD_COVER_COLOR
    overlay_img[mask == 2] = OC_COVER_COLOR

    return img * (1 - alpha) + overlay_img * alpha


def get_contour_images(imgs: list[np.ndarray], masks: list[np.ndarray], preds: list[np.ndarray], **kwargs):
    return [get_contour_image(img, mask, pred, **kwargs) for img, mask, pred in zip(imgs, masks, preds)]


def get_contour_image(img: np.ndarray, mask: np.ndarray = None, pred: np.ndarray = None, class_ids: list[int] = None):
    """Draw contours of the mask and/or prediction over the input image."""
    contour_mask = get_input_image(img.copy(), uint8=True)

    # resize image to match mask size
    other = mask if mask is not None else pred
    if other is not None and img.shape[:2] != other.shape[:2]:
        contour_mask = cv.resize(contour_mask, (other.shape[1], other.shape[0]), interpolation=cv.INTER_NEAREST)

    if class_ids is None:
        class_ids = [1, 2]

    for i, class_id in enumerate(class_ids):
        if mask is not None:
            class_mask = np.where(mask >= class_id, 1, 0).astype(np.uint8)
            contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(contour_mask, contours, -1, TRUE_CONTOUR_COLOR, thickness=1)

        if pred is not None:
            class_mask = np.where(pred >= class_id, 1, 0).astype(np.uint8)
            contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(contour_mask, contours, -1, PRED_CONTOUR_COLOR, thickness=1)

    return contour_mask


def plot_history(hist, figsize=(12, 8), n_cols: int = 4):
    if hist is None:
        return
    used_metrics = sorted([m[6:] for m in hist.keys() if m.startswith('train_')])
    n_metrics = len(used_metrics)

    n_rows = n_metrics // n_cols + int(n_metrics % n_cols > 0)
    _, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = ax.ravel()

    for i, metric in enumerate(used_metrics):
        if f'train_{metric}' in hist:
            ax[i].plot(hist[f'train_{metric}'], label=f'train')
        if f'val_{metric}' in hist:
            ax[i].plot(hist[f'val_{metric}'], label=f'val')
        ax[i].set_title(metric[0].upper() + metric[1:].replace('_', ' '))
        if all([m not in metric for m in ('loss', 'tp', 'tn', 'fp', 'fn')]):
            ax[i].set_ylim(top=1)
        ax[i].legend()

    for ax in ax[n_metrics:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_model_graph(model, input_size, expand_nested=True):
    model_graph = draw_graph(model, input_size=input_size, expand_nested=expand_nested)
    return model_graph.visual_graph


def show_model_view(model, input_size, name='model', fmt='png'):
    x = torch.zeros(input_size, dtype=torch.float32)
    graph = torchviz.make_dot(model(x), params=dict(model.named_parameters()))
    graph.render(name, format=fmt)
    graph.view()


def plot_image_grid(grid: list[list], titles: list[str] | list[list[str]] = None,
                    img_size: int = 3, transpose: bool = False, figsize: tuple = None,
                    save_path: str = None, show: bool = True,
                    mask_legend=None, overlay_legend=None, contour_legend=None):
    rows, cols = len(grid), len(grid[0])

    # swap rows and columns
    if transpose:
        grid = [[grid[j][i] for j in range(rows)] for i in range(cols)]
        rows, cols = cols, rows

    # flatten titles if 2D
    if titles and isinstance(titles[0], list):
        titles = [t for row in titles for t in row]

    # create figure
    if figsize is None:
        figsize = (cols * img_size, rows * img_size)
    _, axes = plt.subplots(rows, cols, sharex='all', sharey='all', figsize=figsize)

    for ax in axes.flatten():
        ax.axis('off')

    # plot images
    i, j = 0, 0
    for idx, ax in enumerate(axes.flatten()):
        if grid[i][j] is not None:
            ax.imshow(grid[i][j])
            ax.axis('on')

        j = (j + 1) % cols
        if j == 0:
            i += 1

        if titles and idx < len(titles):
            ax.set_title(titles[idx])

    if mask_legend:
        axes[rows - 1, mask_legend].legend(handles=[
            Patch(facecolor=BG_COLOR, label='BG'),
            Patch(facecolor=OD_COLOR, label='OD'),
            Patch(facecolor=OC_COLOR, label='OC'),
        ], bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=2)

    if overlay_legend:
        axes[rows - 1, overlay_legend].legend(handles=[
            Patch(color=TP_COLOR / 255, label='TP'),
            Patch(color=FN_COLOR / 255, label='FN'),
            Patch(color=FP_COLOR / 255, label='FP'),
            Patch(color=TN_COLOR / 255, label='TN'),
        ], bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=2)

    if contour_legend:
        axes[rows - 1, contour_legend].legend(handles=[
            Patch(color=np.array(TRUE_CONTOUR_COLOR) / 255, label='True'),
            Patch(color=np.array(PRED_CONTOUR_COLOR) / 255, label='Predicted'),
        ], bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_results(images=None, masks=None, preds=None, types: str | list[str] = None, **kwargs):
    # kwargs: img_size, transpose, figsize, save_path, show
    all_types = [
        'image', 'mask', 'prediction',
        'OD overlay', 'OC overlay',
        'OD cover', 'OC cover',
        'OD contour', 'OC contour', 'contours',
    ]
    if types is None:
        types = ['image', 'mask', 'prediction', 'OD overlay', 'OC overlay']  # default
    elif types == 'all':
        types = all_types
    elif isinstance(types, str):
        types = [types]

    titles = []
    columns = []
    mask_legend_index = None
    overlay_legend_index = None
    contour_legend_index = None

    for i, t in enumerate(types):
        if t == 'image' and images is not None:
            titles.append('Input image')
            col = get_input_images(images)
        elif t == 'mask' and masks is not None:
            titles.append('Ground truth')
            col = get_ground_truth_masks(masks)
            if mask_legend_index is None:
                mask_legend_index = i
        elif t == 'prediction' and preds is not None:
            titles.append('Model prediction')
            col = get_prediction_masks(preds)
            if mask_legend_index is None:
                mask_legend_index = i
        elif t == 'OD overlay' and all(x is not None for x in [masks, preds]):
            titles.append('Optic disc')
            col = get_overlay_images(masks, preds, class_ids=[1, 2])
            if overlay_legend_index is None:
                overlay_legend_index = i
        elif t == 'OC overlay' and all(x is not None for x in [masks, preds]):
            titles.append('Optic cup')
            col = get_overlay_images(masks, preds, class_ids=[2])
            if overlay_legend_index is None:
                overlay_legend_index = i
        elif t == 'OD cover' and all(x is not None for x in [images, masks]):
            titles.append('True mask')
            col = get_cover_images(images, masks)
        elif t == 'OC cover' and all(x is not None for x in [images, preds]):
            titles.append('Predicted mask')
            col = get_cover_images(images, preds)
        elif t == 'contours' and all(x is not None for x in [images, masks, preds]):
            titles.append('Contours')
            col = get_contour_images(images, masks, preds, class_ids=[1, 2])
            if contour_legend_index is None:
                contour_legend_index = i
        elif t == 'OD contour' and all(x is not None for x in [images, masks, preds]):
            titles.append('Optic disc contours')
            col = get_contour_images(images, masks, preds, class_ids=[1])
            if contour_legend_index is None:
                contour_legend_index = i
        elif t == 'OC contour' and all(x is not None for x in [images, masks, preds]):
            titles.append('Optic cup contours')
            col = get_contour_images(images, masks, preds, class_ids=[2])
            if contour_legend_index is None:
                contour_legend_index = i
        else:
            continue
        columns.append(col)

    plot_image_grid(columns, titles=titles, transpose=True, mask_legend=mask_legend_index,
                    overlay_legend=overlay_legend_index, contour_legend=contour_legend_index, **kwargs)


def plot_results_from_loader(mode: str, loader, model, device, n_samples: int = 4,
                             thresh: float = 0.5, od_thresh: float = None, oc_thresh: float = None,
                             class_ids: list = None, base_model=None, **kwargs):
    assert mode in ('binary', 'multiclass', 'multilabel', 'cascade', 'dual')

    if class_ids is None:
        class_ids = [[1, 2]]
    elif isinstance(class_ids, int):
        class_ids = [[class_ids]]
    elif isinstance(class_ids[0], int):
        class_ids = [class_ids]
    tensor_class_ids = torch.tensor(class_ids).to(device)

    # parse keyword arguments
    if 'types' in kwargs:
        types = kwargs['types']
    elif mode == 'binary':
        sign = 'OD' if 1 in class_ids[0] else 'OC'
        types = ['image', 'mask', 'prediction', sign + ' overlay', sign + ' contour']
    elif mode == 'multiclass':
        types = ['image', 'mask', 'prediction', 'OD overlay', 'OC overlay', 'OD contour', 'OC contour']
    elif mode == 'multilabel':
        types = ['image', 'mask', 'prediction', 'OD overlay', 'OC overlay', 'OD contour', 'OC contour']
    elif mode == 'cascade':
        types = ['image', 'mask', 'prediction', 'OD overlay', 'OC overlay', 'OD contour', 'OC contour']
    elif mode == 'dual':
        types = ['image', 'mask', 'prediction', 'OD overlay', 'OC overlay', 'OD contour', 'OC contour']
    else:
        raise ValueError('Invalid model mode: ' + mode)

    img_size = kwargs['img_size'] if 'img_size' in kwargs else 3
    save_path = kwargs['save_path'] if 'save_path' in kwargs else None
    show = kwargs['show'] if 'show' in kwargs else True

    with torch.no_grad():
        samples_so_far = 0
        images_all, masks_all, preds_all = [], [], []

        for images, masks in loader:
            preds, *_ = predict(
                mode, model, images, masks, device, thresh, od_thresh, oc_thresh,
                base_model=base_model, binary_labels=tensor_class_ids,
            )

            if mode == 'binary':  # Binarize masks and predictions
                masks = masks.to(device).long()
                masks = torch.where(torch.isin(masks, tensor_class_ids), 1, 0)
                if class_ids == [[2]]:
                    preds[preds == 1] = 2
                    masks[masks == 1] = 2

            images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            masks = masks.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            images_all.append(images)
            masks_all.append(masks)
            preds_all.append(preds)

            samples_so_far += len(images)
            if samples_so_far >= n_samples:
                break

        images_all = np.concatenate(images_all, axis=0)
        masks_all = np.concatenate(masks_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)

        images = images_all[:n_samples]
        masks = masks_all[:n_samples]
        preds = preds_all[:n_samples]

    plot_results(
        images, masks, preds, img_size=img_size, save_path=save_path, show=show, types=types
    )
