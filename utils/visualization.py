import matplotlib.pyplot as plt
import numpy as np
import torch
import torchviz
from torchview import draw_graph
from matplotlib.patches import Patch


def show_model_graph(model, input_size, expand_nested=True):
    model_graph = draw_graph(model, input_size=input_size, expand_nested=expand_nested)
    return model_graph.visual_graph


def show_model_view(model, input_size, name='model', fmt='png'):
    x = torch.zeros(input_size, dtype=torch.float32)
    graph = torchviz.make_dot(model(x), params=dict(model.named_parameters()))
    graph.render(name, format=fmt)
    graph.view()


def unnormalize(image, mean, std):
    """
    Restore a normalized image to its original state.
    """
    restored_image = image.copy()
    for c in range(restored_image.shape[2]):
        restored_image[:, :, c] = (restored_image[:, :, c] * std[c]) + mean[c]
    return restored_image


tp_color = np.array((0, 1, 0), dtype=np.uint8) * 255
tn_color = np.array((0, 0, 0), dtype=np.uint8) * 255
fp_color = np.array((1, 0, 0), dtype=np.uint8) * 255
fn_color = np.array((1, 1, 0), dtype=np.uint8) * 255

od_color = np.array((0, 0, 1))
oc_color = np.array((0, 1, 1))


def get_input_images(images):
    return [get_input_image(img) for img in images]


def get_input_image(img):
    if img.max() > 1:
        img = img / 255
    return img


def get_cover_images(masks, preds, **kwargs):
    return [get_cover_image(mask, pred, **kwargs) for mask, pred in zip(preds, masks)]


def get_cover_image(mask, pred, class_ids: list[int] = None):
    """
    Visualize the correctness of the segmentation. The results are color-coded as follows:
    True Positive: green
    True Negative: black
    False Positive: red
    False Negative: yellow
    """
    cover = np.zeros((*mask.shape, 3), dtype=np.uint8)

    if class_ids is None:
        # treat image as binary (0 - background, non-zero - foreground)
        cover[(mask != 0) & (pred != 0)] = tp_color
        cover[(mask == 0) & (pred == 0)] = tn_color
        cover[(mask == 0) & (pred != 0)] = fp_color
        cover[(mask != 0) & (pred == 0)] = fn_color
    else:
        # True positive (TP): both mask and pred have one of the class_ids at the same pixel
        tp_mask = np.isin(mask, class_ids) & np.isin(mask, pred)
        cover[tp_mask] = tp_color

        # True negative (TN): neither mask nor pred have one of the class_ids at the same position
        tn_mask = np.logical_not(np.isin(mask, class_ids)) & np.logical_not(np.isin(pred, class_ids))
        cover[tn_mask] = tn_color

        # False positive (FP): pred has one of the class_ids, but mask does not
        fp_mask = np.logical_not(np.isin(mask, class_ids)) & np.isin(pred, class_ids)
        cover[fp_mask] = fp_color

        # False negative (FN): pred does not have one of the class_ids, but mask does
        fn_mask = np.isin(mask, class_ids) & np.logical_not(np.isin(pred, class_ids))
        cover[fn_mask] = fn_color

    return cover


def get_overlay_images(imgs, masks, **kwargs):
    return [get_overlay_image(img, mask, **kwargs) for img, mask in zip(imgs, masks)]


def get_overlay_image(img, mask, alpha=0.3):
    img = get_input_image(img)

    overlay = np.zeros_like(img)
    overlay[mask == 1] = od_color
    overlay[mask == 2] = oc_color

    return img * (1 - alpha) + overlay * alpha


def plot_image_grid(grid: list[list], titles: list[str] | list[list[str]] = None,
                    img_size: int = 3, transpose: bool = False, figsize: tuple = None,
                    save_path: str = None, show: bool = True,
                    mask_legend=None, cover_legend=None, contour_legend=None):
    rows, cols = len(grid), len(grid[0])

    # swap rows and cols
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
            Patch(facecolor=(0.27, 0.01, 0.33), label='BG'),
            Patch(facecolor=(0.13, 0.56, 0.55), label='OD'),
            Patch(facecolor=(0.99, 0.91, 0.14), label='OC'),
        ], bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)

    if cover_legend:
        axes[rows - 1, cover_legend].legend(handles=[
            Patch(color=tp_color / 255, label='TP'),
            Patch(color=fn_color / 255, label='FN'),
            Patch(color=fp_color / 255, label='FP'),
            Patch(color=tn_color / 255, label='TN'),
        ], bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)

    if contour_legend:
        axes[rows - 1, contour_legend].legend(handles=[
            Patch(color=od_color, label='T-OD'),
            Patch(color=od_color, label='P-OD'),
            Patch(color=oc_color, label='T-OC'),
            Patch(color=oc_color, label='P-OC'),
        ], bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_side_by_side(images=None, masks=None, preds=None, types: str | list[str] = None, **kwargs):
    if types is None:
        types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover']
    elif types == 'all':
        types = ['image', 'mask', 'prediction', 'OD cover', 'OC cover', 'OD overlay', 'OC overlay', 'contours']
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
            col = masks
            if mask_legend_index is None:
                mask_legend_index = i
        elif t == 'prediction' and preds is not None:
            titles.append('Model prediction')
            col = preds
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
            col = [None] * len(images)  # TODO
            if contour_legend_index is None:
                contour_legend_index = i
        else:
            continue
        columns.append(col)

    plot_image_grid(columns, titles=titles, transpose=True, mask_legend=mask_legend_index,
                    cover_legend=cover_legend_index, contour_legend=contour_legend_index, **kwargs)


def plot_results_from_loader(loader, model, device: str = 'cuda', n_samples: int = 4, save_path=None, show=True):
    model.eval()
    model = model.to(device=device)
    with torch.no_grad():
        batch = next(iter(loader))
        images, masks = batch

        images = images.float().to(device=device)
        masks = masks.long().to(device=device)

        outputs = model(images)
        # softmax not needed because index of max value is the same before and after calling softmax
        preds = torch.argmax(outputs, dim=1)

        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()

        images = images[:n_samples]
        masks = masks[:n_samples]
        preds = preds[:n_samples]

    plot_side_by_side(
        images, masks, preds, img_size=3, save_path=save_path, show=show,
        types=['image', 'mask', 'prediction', 'OD cover', 'OC cover']
    )


def plot_results(images, masks, preds, save_path=None, show=True):
    plot_side_by_side(
        images, masks, preds, img_size=3, save_path=save_path, show=show,
        types=['image', 'mask', 'prediction', 'OD cover', 'OC cover']
    )
