import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


def restore_image(image, mean, std):
    """
    Restore a normalized image to its original state.
    """
    restored_image = image.copy()
    for c in range(restored_image.shape[2]):
        restored_image[:, :, c] = (restored_image[:, :, c] * std[c]) + mean[c]
    return restored_image


# plot segmentation results
def plot_side_by_side_results(images, masks, preds, save_path=None, show=True):
    """
    Visualize the segmentation results in 3 columns as follows:
    1st column: input image
    2nd column: ground truth
    3rd column: model prediction
    """
    fig, ax = plt.subplots(len(images), 3, figsize=(8, 3 * len(images)))

    for i, (image, mask, pred) in enumerate(zip(images, masks, preds)):
        ax[i, 0].imshow(image)
        ax[i, 0].set_title('Image')

        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Ground truth')

        ax[i, 2].imshow(pred)
        ax[i, 2].set_title('Prediction')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_overlaid_results(images, masks, preds, save_path=None, show=True,
                          alpha=0.4, od_color=(0, 0, 1), oc_color=(0, 1, 1)):
    """
    Visualize the segmentation results in 2 columns as follows:
    1st column: ground truth overlaid on the input image
    2nd column: model prediction overlaid on the input image
    """
    fig, ax = plt.subplots(len(images), 2, figsize=(6, 3 * len(images)))

    for i, (image, mask, pred) in enumerate(zip(images, masks, preds)):
        overlay_mask = np.zeros_like(image)
        overlay_mask[mask == 1] = od_color
        overlay_mask[mask == 2] = oc_color
        overlay_mask = image * (1 - alpha) + overlay_mask * alpha
        ax[i, 0].imshow(overlay_mask)
        ax[i, 0].set_title('Ground truth')

        overlay_pred = np.zeros_like(image)
        overlay_pred[pred == 1] = od_color
        overlay_pred[pred == 2] = oc_color
        overlay_pred = image * (1 - alpha) + overlay_pred * alpha
        ax[i, 1].imshow(overlay_pred)
        ax[i, 1].set_title('Prediction')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_correct_results(images, masks, preds, save_path=None, show=True,
                         tp_color=(0, 1, 0), tn_color=(0, 0, 0),
                         fp_color=(1, 0, 0), fn_color=(1, 1, 0)):
    """
    Visualize the correctness of the segmentation. The results are color-coded as follows:
    True Positive: green
    True Negative: black
    False Positive: red
    False Negative: yellow

    The results are displayed in 3 columns as follows:
    1st column: input image
    2nd column: color-coded correct pixels in the optic disc compared to the ground truth
    3rd column: color-coded correct pixels in the optic cup compared to the ground truth
    """
    fig, ax = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)))
    for i, (image, mask, pred) in enumerate(zip(images, masks, preds)):
        ax[i, 0].imshow(image)
        ax[i, 0].set_title('Image')

        correct_od = np.zeros_like(image)
        correct_od[(mask != 0) & (pred != 0)] = tp_color
        correct_od[(mask == 0) & (pred == 0)] = tn_color
        correct_od[(mask == 0) & (pred != 0)] = fp_color
        correct_od[(mask != 0) & (pred == 0)] = fn_color
        ax[i, 1].imshow(correct_od)
        ax[i, 1].set_title('Optic disc')

        correct_oc = np.zeros_like(image)
        correct_oc[(mask == 2) & (pred == 2)] = tp_color
        correct_oc[(mask != 2) & (pred != 2)] = tn_color
        correct_oc[(mask != 2) & (pred == 2)] = fp_color
        correct_oc[(mask == 2) & (pred != 2)] = fn_color
        ax[i, 2].imshow(correct_oc)
        ax[i, 2].set_title('Optic cup')

    legend_patches = [
        Patch(color=tp_color, label='True Positive'),
        Patch(color=fn_color, label='False Negative'),
        Patch(color=fp_color, label='False Positive'),
        Patch(color=tn_color, label='True Negative'),
    ]
    ax[3, 1].legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_results_from_loader(loader, model, device, save_path=None, show=True, mean=None, std=None):
    model.eval()
    model = model.to(device=device)
    with torch.no_grad():
        batch = next(iter(loader))
        images, masks = batch

        images = images.to(device=device)
        masks = masks.to(device=device)

        outputs = model(images)
        # softmax not needed because index of max value is the same before and after calling softmax
        preds = torch.argmax(outputs, dim=1)

        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()

    plot_results(images, masks, preds, save_path=save_path, show=show, mean=mean, std=std)


def plot_results(images, masks, preds, save_path=None, show=True, mean=None, std=None):
    """
    Visualize the segmentation results in 5 columns as follows:
    1st column: input image
    2nd column: ground truth
    3rd column: model prediction
    4th column: color-coded correct pixels in the OD compared to the GT
    5th column: color-coded correct pixels in the OC compared to the GT
    """
    tp_color = (0, 1, 0)
    tn_color = (0, 0, 0)
    fp_color = (1, 0, 0)
    fn_color = (1, 1, 0)

    fig, ax = plt.subplots(len(images), 5, figsize=(15, 3 * len(images)))
    for i, (image, mask, pred) in enumerate(zip(images, masks, preds)):
        # un-normalize image with mean and std if necessary
        if mean is not None and std is not None:
            image = restore_image(image, mean, std)

        ax[i, 0].imshow(image)
        if i == 0:
            ax[i, 0].set_title('Input image')

        ax[i, 1].imshow(mask)
        if i == 0:
            ax[i, 1].set_title('Ground truth')

        ax[i, 2].imshow(pred)
        if i == 0:
            ax[i, 2].set_title('Model prediction')

        correct_od = np.zeros_like(image)
        correct_od[(mask != 0) & (pred != 0)] = tp_color
        correct_od[(mask == 0) & (pred == 0)] = tn_color
        correct_od[(mask == 0) & (pred != 0)] = fp_color
        correct_od[(mask != 0) & (pred == 0)] = fn_color
        ax[i, 3].imshow(correct_od)
        if i == 0:
            ax[i, 3].set_title('Optic disc')

        correct_oc = np.zeros_like(image)
        correct_oc[(mask == 2) & (pred == 2)] = tp_color
        correct_oc[(mask != 2) & (pred != 2)] = tn_color
        correct_oc[(mask != 2) & (pred == 2)] = fp_color
        correct_oc[(mask == 2) & (pred != 2)] = fn_color
        ax[i, 4].imshow(correct_oc)
        if i == 0:
            ax[i, 4].set_title('Optic cup')

    legend_class_patches = [
        Patch(facecolor=(0.27, 0.01, 0.33), label='Background (BG)'),
        Patch(facecolor=(0.13, 0.56, 0.55), label='Optic disc (OD)'),
        Patch(facecolor=(0.99, 0.91, 0.14), label='Optic cup (OC)'),
    ]
    ax[3, 1].legend(handles=legend_class_patches, bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=1)

    legend_color_patches = [
        Patch(color=tp_color, label='True Positive'),
        Patch(color=fn_color, label='False Negative'),
        Patch(color=fp_color, label='False Positive'),
        Patch(color=tn_color, label='True Negative'),
    ]
    ax[3, 3].legend(handles=legend_color_patches, bbox_to_anchor=(0.5, -0.6), loc='lower center', ncol=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_correct_results_with_more_colors(images, masks, preds, save_path=None, show=True):
    """
    Visualize the results of the segmentation with different colors for each case (9 colors in total).
    This function is not recommended, because it uses too many colors and it is difficult to understand the results.
    """
    black = (0, 0, 0)
    red = (1, 0, 0)
    green = (0, 1, 0)
    blue = (0, 0, 1)
    cyan = (0, 1, 1)
    magenta = (1, 0, 1)
    yellow = (1, 1, 0)
    white = (1, 1, 1)
    gray = (0.5, 0.5, 0.5)

    fig, ax = plt.subplots(len(images), 2, figsize=(6, 3 * len(images)))

    for i, (image, mask, pred) in enumerate(zip(images, masks, preds)):
        ax[i, 0].imshow(image)
        ax[i, 0].set_title('Image')

        correct = np.zeros_like(image)
        correct[(mask == 0) & (pred == 0)] = black  # true BG
        correct[(mask == 0) & (pred == 1)] = yellow  # should be BG, but predicted OD
        correct[(mask == 0) & (pred == 2)] = cyan  # should be BG, but predicted OC

        correct[(mask == 1) & (pred == 1)] = green  # true OD
        correct[(mask == 1) & (pred == 2)] = white  # should be OD, but predicted OC
        correct[(mask == 1) & (pred == 0)] = red  # should be OD, but predicted BG

        correct[(mask == 2) & (pred == 2)] = blue  # true OC
        correct[(mask == 2) & (pred == 1)] = magenta  # should be OC, but predicted OD
        correct[(mask == 2) & (pred == 0)] = gray  # should be OC, but predicted BG

        ax[i, 1].imshow(correct)
        ax[i, 1].set_title('Correctness')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
