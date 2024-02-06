import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchviz
from torchview import draw_graph
from matplotlib.patches import Patch

__all__ = [
    'show_model_graph', 'show_model_view', 'plot_history', 'unnormalize', 'get_input_images', 'get_input_image',
    'get_ground_truth_masks', 'get_ground_truth_mask', 'get_prediction_masks', 'get_prediction_mask',
    'get_cover_images', 'get_cover_image', 'get_overlay_images', 'get_overlay_image', 'get_contour_images',
    'get_contour_image', 'plot_image_grid', 'plot_results', 'plot_side_by_side', 'plot_results_from_loader',
]


def show_model_graph(model, input_size, expand_nested=True):
    model_graph = draw_graph(model, input_size=input_size, expand_nested=expand_nested)
    return model_graph.visual_graph


def show_model_view(model, input_size, name='model', fmt='png'):
    x = torch.zeros(input_size, dtype=torch.float32)
    graph = torchviz.make_dot(model(x), params=dict(model.named_parameters()))
    graph.render(name, format=fmt)
    graph.view()


def plot_history(h, figsize=(14, 8)):
    if h is None:
        return
    used_metrics = sorted([m[6:] for m in h.keys() if m.startswith('train_')])

    n = len(used_metrics)
    _, ax = plt.subplots(n // 4 + 1, 4, figsize=figsize)
    ax = ax.ravel()

    for i, metric in enumerate(used_metrics):
        ax[i].plot(h[f'train_{metric}'], label=f'train')
        ax[i].plot(h[f'val_{metric}'], label=f'val')
        ax[i].set_title(metric[0].upper() + metric[1:].replace('_', ' '))
        if all([m not in metric for m in ('loss', 'tp', 'tn', 'fp', 'fn')]):
            ax[i].set_ylim(top=1)
        ax[i].legend()

    for ax in ax[len(used_metrics):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


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
tp_color = np.array((0, 1, 0), dtype=np.uint8) * 255  # green
tn_color = np.array((0, 0, 0), dtype=np.uint8) * 255  # black
fp_color = np.array((1, 0, 0), dtype=np.uint8) * 255  # red
fn_color = np.array((1, 1, 0), dtype=np.uint8) * 255  # yellow

# Overlay colors
od_overlay_color = np.array((0, 0, 1))  # blue
oc_overlay_color = np.array((0, 1, 1))  # cyan

# Contour colors
true_color = (0, 0, 255)  # blue
predicted_color = (0, 0, 0)  # black


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


def get_contour_image(img, mask=None, pred=None, class_ids: list[int] = None, colors: list[tuple] = None):
    contour_mask = img.copy()

    # resize image to match mask size
    other = mask if mask is not None else pred
    if other is not None and img.shape[:2] != other.shape[:2]:
        contour_mask = cv.resize(contour_mask, (other.shape[1], other.shape[0]), interpolation=cv.INTER_NEAREST)

    if class_ids is None:
        class_ids = [1, 2]

    for i, class_id in enumerate(class_ids):
        color = colors[i] if colors is not None and i < len(colors) else true_color

        if mask is not None:
            class_mask = np.where(mask >= class_id, 1, 0).astype(np.uint8)
            contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(contour_mask, contours, -1, color, thickness=1)

        if pred is not None:
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


plot_side_by_side = plot_results  # alias


def plot_results_from_loader(mode: str, loader, model, device: str = 'cuda', n_samples: int = 4, thresh: float = 0.5,
                             class_ids: list = None, model0=None, **kwargs):
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
        types = ['image', 'mask', 'prediction', sign + ' cover', sign + ' contour']
    elif mode == 'multiclass':
        types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover', 'OD contour', 'OC contour']
    elif mode == 'multilabel':
        types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover', 'OD contour', 'OC contour']
    elif mode == 'cascade':
        types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover', 'OD contour', 'OC contour']
    elif mode == 'dual':
        types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover', 'OD contour', 'OC contour']
    else:
        raise ValueError('Invalid model mode: ' + mode)

    img_size = kwargs['img_size'] if 'img_size' in kwargs else 3
    save_path = kwargs['save_path'] if 'save_path' in kwargs else None
    show = kwargs['show'] if 'show' in kwargs else True

    model.eval()
    model = model.to(device)
    if model0 is not None:
        model0.eval()
        model0 = model0.to(device)

    with torch.no_grad():
        samples_so_far = 0
        images_all, masks_all, preds_all = [], [], []

        for images, masks in loader:
            images = images.float().to(device)
            masks = masks.long().to(device)

            if mode == 'multiclass':
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            elif mode == 'multilabel':
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = torch.zeros_like(probs[:, 0])
                for i in range(1, probs.shape[1]):
                    preds += (probs[:, i] > thresh).long()

            elif mode == 'binary':
                masks = torch.where(torch.isin(masks, tensor_class_ids), 1, 0)

                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > thresh).squeeze(1).long()

                if class_ids == [[2]]:
                    preds[preds == 1] = 2
                    masks[masks == 1] = 2

            elif mode == 'cascade':
                od_outputs = model0(images)
                od_probs = torch.sigmoid(od_outputs)
                od_preds = (od_probs > thresh).long()

                cropped_images = images * od_preds

                oc_outputs = model(cropped_images)
                oc_probs = torch.sigmoid(oc_outputs)
                oc_preds = (oc_probs > thresh).long()

                # Combine OD and OC predictions
                preds = torch.zeros_like(od_preds)
                preds[od_preds == 1] = 1
                preds[oc_preds == 1] = 2
                preds = preds.squeeze(1)

            elif mode == 'dual':
                od_outputs, oc_outputs = model(images)

                od_probs = torch.sigmoid(od_outputs)
                od_preds = (od_probs > thresh).long()

                oc_probs = torch.sigmoid(oc_outputs)
                oc_preds = (oc_probs > thresh).long()

                # Combine OD and OC predictions
                preds = torch.zeros_like(od_preds)
                preds[od_preds == 1] = 1
                preds[oc_preds == 1] = 2
                preds = preds.squeeze(1)

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
