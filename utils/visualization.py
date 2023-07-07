import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchviz
from torchview import draw_graph
from matplotlib.patches import Patch

from utils.metrics import get_best_OD_examples, get_worst_OD_examples, get_best_OC_examples, get_worst_OC_examples


def show_model_graph(model, input_size, expand_nested=True):
    model_graph = draw_graph(model, input_size=input_size, expand_nested=expand_nested)
    return model_graph.visual_graph


def show_model_view(model, input_size, name='model', fmt='png'):
    x = torch.zeros(input_size, dtype=torch.float32)
    graph = torchviz.make_dot(model(x), params=dict(model.named_parameters()))
    graph.render(name, format=fmt)
    graph.view()


def unnormalize(image, mean, std):
    """Restore a normalized image to its original state."""
    restored_image = image.copy()
    for c in range(restored_image.shape[-1]):
        restored_image[:, :, c] = (restored_image[:, :, c] * std[c]) + mean[c]
    return restored_image


# Label colors
bg_color = np.array((0.27, 0.01, 0.33), dtype=np.float32)
od_color = np.array((0.13, 0.56, 0.55), dtype=np.float32)
oc_color = np.array((0.99, 0.91, 0.14), dtype=np.float32)

# Cover colors
tp_color = np.array((0, 1, 0), dtype=np.uint8) * 255
tn_color = np.array((0, 0, 0), dtype=np.uint8) * 255
fp_color = np.array((1, 0, 0), dtype=np.uint8) * 255
fn_color = np.array((1, 1, 0), dtype=np.uint8) * 255

# Overlay colors
od_overlay_color = np.array((0, 0, 1))
oc_overlay_color = np.array((0, 1, 1))

# Contour colors
true_color = (0, 0, 255)
predicted_color = (0, 0, 0)


def get_input_images(imgs):
    return [get_input_image(img) for img in imgs]


def get_input_image(img):
    if img.max() > 1:
        img = img / 255
    return img


def get_ground_truth_masks(masks):
    return [get_ground_truth_mask(mask) for mask in masks]


def get_ground_truth_mask(mask):
    gt = np.zeros((*mask.shape, 3), dtype=np.float32)
    gt[mask == 0] = bg_color
    gt[mask == 1] = od_color
    gt[mask == 2] = oc_color
    return gt


def get_prediction_masks(preds):
    return [get_prediction_mask(pred) for pred in preds]


def get_prediction_mask(pred):
    pr = np.zeros((*pred.shape, 3), dtype=np.float32)
    pr[pred == 0] = bg_color
    pr[pred == 1] = od_color
    pr[pred == 2] = oc_color
    return pr


def get_cover_images(masks, preds, **kwargs):
    return [get_cover_image(mask, pred, **kwargs) for mask, pred in zip(masks, preds)]


def get_cover_image(mask, pred, class_ids: list[int] = None):
    """
    Visualize the correctness of the segmentation. The results are color-coded as follows:
    True Positive: green
    True Negative: black
    False Positive: red
    False Negative: yellow
    """
    cover_img = np.zeros((*mask.shape, 3), dtype=np.uint8)

    if class_ids is None:
        # treat image as binary (0 - background, non-zero - foreground)
        cover_img[(mask != 0) & (pred != 0)] = tp_color
        cover_img[(mask == 0) & (pred == 0)] = tn_color
        cover_img[(mask == 0) & (pred != 0)] = fp_color
        cover_img[(mask != 0) & (pred == 0)] = fn_color
    else:
        # True positive (TP): both mask and pred have one of the class_ids at the same pixel
        tp_mask = np.isin(mask, class_ids) & np.isin(pred, class_ids)
        cover_img[tp_mask] = tp_color

        # True negative (TN): neither mask nor pred have one of the class_ids at the same position
        tn_mask = np.logical_not(np.isin(mask, class_ids)) & np.logical_not(np.isin(pred, class_ids))
        cover_img[tn_mask] = tn_color

        # False positive (FP): pred has one of the class_ids, but mask does not
        fp_mask = np.logical_not(np.isin(mask, class_ids)) & np.isin(pred, class_ids)
        cover_img[fp_mask] = fp_color

        # False negative (FN): pred does not have one of the class_ids, but mask does
        fn_mask = np.isin(mask, class_ids) & np.logical_not(np.isin(pred, class_ids))
        cover_img[fn_mask] = fn_color

    return cover_img


def get_overlay_images(imgs, masks, **kwargs):
    return [get_overlay_image(img, mask, **kwargs) for img, mask in zip(imgs, masks)]


def get_overlay_image(img, mask, alpha=0.3):
    img = get_input_image(img)

    overlay_img = np.zeros_like(img)
    overlay_img[mask == 1] = od_overlay_color
    overlay_img[mask == 2] = oc_overlay_color

    return img * (1 - alpha) + overlay_img * alpha


def get_contour_images(imgs, masks, preds, **kwargs):
    return [get_contour_image(img, mask, pred, **kwargs) for img, mask, pred in zip(imgs, masks, preds)]


def get_contour_image(img, mask, pred, class_ids: list[int] = None):
    contour_mask = img.copy()

    if class_ids is None:
        class_ids = [1, 2]

    for class_id in class_ids:
        class_mask = np.where(mask >= class_id, 1, 0).astype(np.uint8)
        contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(contour_mask, contours, -1, true_color, thickness=1)

        class_mask = np.where(pred >= class_id, 1, 0).astype(np.uint8)
        contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(contour_mask, contours, -1, predicted_color, thickness=1)

    return get_input_image(contour_mask)


def plot_image_grid(grid: list[list], titles: list[str] | list[list[str]] = None,
                    img_size: int = 3, transpose: bool = False, figsize: tuple = None,
                    save_path: str = None, show: bool = True,
                    mask_legend=None, cover_legend=None, contour_legend=None):
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
            Patch(facecolor=bg_color, label='BG'),
            Patch(facecolor=od_color, label='OD'),
            Patch(facecolor=oc_color, label='OC'),
        ], bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=2)

    if cover_legend:
        axes[rows - 1, cover_legend].legend(handles=[
            Patch(color=tp_color / 255, label='TP'),
            Patch(color=fn_color / 255, label='FN'),
            Patch(color=fp_color / 255, label='FP'),
            Patch(color=tn_color / 255, label='TN'),
        ], bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=2)

    if contour_legend:
        axes[rows - 1, contour_legend].legend(handles=[
            Patch(color=np.array(true_color) / 255, label='True'),
            Patch(color=np.array(predicted_color) / 255, label='Predicted'),
        ], bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_side_by_side(*args, **kwargs):
    plot_results(*args, **kwargs)


def plot_results(images=None, masks=None, preds=None, types: str | list[str] = None, **kwargs):
    # kwargs: img_size, transpose, figsize, save_path, show

    if types is None:
        types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover']
    elif types == 'all':
        types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover',
                 'OD overlay', 'OC overlay', 'OD contour', 'OC contour', 'contours']
    elif isinstance(types, str):
        types = [types]

    titles = []
    columns = []
    mask_legend_index = None
    cover_legend_index = None
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
        elif t == 'OD cover' and all(x is not None for x in [masks, preds]):
            titles.append('Optic disc')
            col = get_cover_images(masks, preds, class_ids=[1, 2])
            if cover_legend_index is None:
                cover_legend_index = i
        elif t == 'OC cover' and all(x is not None for x in [masks, preds]):
            titles.append('Optic cup')
            col = get_cover_images(masks, preds, class_ids=[2])
            if cover_legend_index is None:
                cover_legend_index = i
        elif t == 'OD overlay' and all(x is not None for x in [images, masks]):
            titles.append('True mask')
            col = get_overlay_images(images, masks)
        elif t == 'OC overlay' and all(x is not None for x in [images, preds]):
            titles.append('Predicted mask')
            col = get_overlay_images(images, preds)
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
                    cover_legend=cover_legend_index, contour_legend=contour_legend_index, **kwargs)


def plot_results_from_loader(loader, model, device: str = 'cuda', n_samples: int = 4, **kwargs):
    # kwargs: save_path, show, types, img_size

    model.eval()
    model = model.to(device=device)

    with torch.no_grad():
        samples_so_far = 0
        images_all, masks_all, preds_all = [], [], []

        for images, masks in loader:
            images = images.float().to(device=device)
            masks = masks.long().to(device=device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

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

    types = kwargs['types'] if 'types' in kwargs else ['image', 'mask', 'prediction', 'OD cover', 'OC cover']
    img_size = kwargs['img_size'] if 'img_size' in kwargs else 3
    save_path = kwargs['save_path'] if 'save_path' in kwargs else None
    show = kwargs['show'] if 'show' in kwargs else True

    plot_results(
        images, masks, preds, img_size=img_size, save_path=save_path, show=show, types=types
    )


def plot_best_OD_examples(model, loader, n: int = 4, **kwargs):
    examples = get_best_OD_examples(model, loader, n)

    images = [e[0] for e in examples]
    masks = [e[1] for e in examples]
    preds = [e[2] for e in examples]

    plot_results(images, masks, preds, **kwargs)


def plot_worst_OD_examples(model, loader, n: int = 4, **kwargs):
    examples = get_worst_OD_examples(model, loader, n)

    images = [e[0] for e in examples]
    masks = [e[1] for e in examples]
    preds = [e[2] for e in examples]

    plot_results(images, masks, preds, **kwargs)


def plot_best_OC_examples(model, loader, n: int = 4, **kwargs):
    examples = get_best_OC_examples(model, loader, n)

    images = [e[0] for e in examples]
    masks = [e[1] for e in examples]
    preds = [e[2] for e in examples]

    plot_results(images, masks, preds, **kwargs)


def plot_worst_OC_examples(model, loader, n: int = 4, **kwargs):
    examples = get_worst_OC_examples(model, loader, n)

    images = [e[0] for e in examples]
    masks = [e[1] for e in examples]
    preds = [e[2] for e in examples]

    plot_results(images, masks, preds, **kwargs)
