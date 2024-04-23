import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from modules.inference import predict
from modules.metrics import calculate_metrics, get_metrics
from modules.preprocessing import inverse_polar_transform
from modules.postprocessing import interprocess, postprocess

__all__ = [
    'undo_polar_transform', 'get_threshold_stats',
    'plot_curves', 'roc_curve', 'precision_recall_curve',
    'threshold_tuning_curve',
]


def undo_polar_transform(images_batch: torch.Tensor, masks_batch: torch.Tensor, preds_batch: torch.Tensor):
    np_images = images_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    np_masks = masks_batch.detach().cpu().numpy()
    np_preds = preds_batch.detach().cpu().numpy()

    new_images = np.zeros_like(np_images)
    new_masks = np.zeros_like(np_masks)
    new_preds = np.zeros_like(np_preds)

    for i, _ in enumerate(np_images):
        new_images[i] = inverse_polar_transform(np_images[i])
        new_masks[i] = inverse_polar_transform(np_masks[i])
        new_preds[i] = inverse_polar_transform(np_preds[i])

    images_batch = torch.from_numpy(new_images.transpose(0, 3, 1, 2)).float().to(images_batch.device)
    masks_batch = torch.from_numpy(new_masks).long().to(masks_batch.device)
    preds_batch = torch.from_numpy(new_preds).long().to(preds_batch.device)

    return images_batch, masks_batch, preds_batch


def get_threshold_stats(mode, model, loader, base_model=None):
    data = []

    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            preds, probs, loss = predict(
                mode, model, images, masks, base_model=base_model,
                post_process_fn=postprocess,
                inter_process_fn=interprocess if mode == 'cascade' else None,
            )

            od_probs = probs[:, 0, :, :]
            oc_probs = probs[:, 1, :, :]
            data.append((images, masks, od_probs, oc_probs))

    thresholds = np.linspace(0, 1, 101)

    tp_od = np.zeros_like(thresholds)
    tn_od = np.zeros_like(thresholds)
    fp_od = np.zeros_like(thresholds)
    fn_od = np.zeros_like(thresholds)

    tp_oc = np.zeros_like(thresholds)
    tn_oc = np.zeros_like(thresholds)
    fp_oc = np.zeros_like(thresholds)
    fn_oc = np.zeros_like(thresholds)

    for i, thresh in enumerate(thresholds):
        for images, masks, od_probs, oc_probs in data:
            od_preds = (od_probs > thresh).long()
            oc_preds = (oc_probs > thresh).long()

            preds = torch.zeros_like(oc_preds)
            preds[od_preds == 1] = 1
            preds[oc_preds == 1] = 2
            met = get_metrics(masks, preds, [[1, 2], [2]])

            tp_od[i] += met['tp_OD']
            tn_od[i] += met['tn_OD']
            fp_od[i] += met['fp_OD']
            fn_od[i] += met['fn_OD']

            tp_oc[i] += met['tp_OC']
            tn_oc[i] += met['tn_OC']
            fp_oc[i] += met['fp_OC']
            fn_oc[i] += met['fn_OC']

    metrics = defaultdict(lambda: np.zeros_like(thresholds))

    for i, (tp, tn, fp, fn) in enumerate(zip(tp_od, tn_od, fp_od, fn_od)):
        met = calculate_metrics(tp, tn, fp, fn)
        for k, v in met.items():
            metrics[f'{k}_OD'][i] = v

    for i, (tp, tn, fp, fn) in enumerate(zip(tp_oc, tn_oc, fp_oc, fn_oc)):
        met = calculate_metrics(tp, tn, fp, fn)
        for k, v in met.items():
            metrics[f'{k}_OC'][i] = v

    df = pd.DataFrame({'threshold': thresholds, **metrics})

    df['precision_OD'][100] = 1
    df['precision_OC'][100] = 1

    df['gmean_OD'] = np.sqrt(df['sensitivity_OD'] * df['specificity_OD'])
    df['gmean_OC'] = np.sqrt(df['sensitivity_OC'] * df['specificity_OC'])

    return df


def threshold_tuning_curve(df, tuning_metric: str = 'dice'):
    max_od_index = df[f'{tuning_metric}_OD'].idxmax()
    max_oc_index = df[f'{tuning_metric}_OC'].idxmax()

    best_threshold_od_tune = df['threshold'][max_od_index]
    best_threshold_oc_tune = df['threshold'][max_oc_index]

    plt.figure(figsize=(9, 6))
    plt.plot(df['threshold'], df[f'{tuning_metric}_OD'], label=f'OD {tuning_metric}')
    plt.plot(df['threshold'], df[f'{tuning_metric}_OC'], label=f'OC {tuning_metric}')
    plt.scatter(best_threshold_od_tune, df[f'{tuning_metric}_OD'][max_od_index], c='b',
                label=f'Best OD threshold: {best_threshold_od_tune:.2f}', zorder=3)
    plt.scatter(best_threshold_oc_tune, df[f'{tuning_metric}_OC'][max_oc_index], c='r',
                label=f'Best OC threshold: {best_threshold_oc_tune:.2f}', zorder=3)
    plt.xlabel('Threshold')
    plt.ylabel(tuning_metric.capitalize())
    plt.title(f'Threshold Tuning Curve for {tuning_metric.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_threshold_od_tune, best_threshold_oc_tune


def roc_curve(df):
    od_distances = np.sqrt(df['fpr_OD'] ** 2 + (1 - df['sensitivity_OD']) ** 2)
    oc_distances = np.sqrt(df['fpr_OC'] ** 2 + (1 - df['sensitivity_OC']) ** 2)

    min_index_od = od_distances.idxmin()
    min_index_oc = oc_distances.idxmin()

    best_threshold_od_roc = df['threshold'][min_index_od]
    best_threshold_oc_roc = df['threshold'][min_index_oc]

    plt.figure(figsize=(9, 6))
    plt.plot(df['fpr_OD'], df['sensitivity_OD'], label='OD curve')
    plt.plot(df['fpr_OC'], df['sensitivity_OC'], label='OC curve')
    plt.scatter(df['fpr_OD'][min_index_od], df['sensitivity_OD'][min_index_od], c='b',
                label=f'Best OD threshold: {best_threshold_od_roc:.2f}', zorder=3)
    plt.scatter(df['fpr_OC'][min_index_oc], df['sensitivity_OC'][min_index_oc], c='r',
                label=f'Best OC threshold: {best_threshold_oc_roc:.2f}', zorder=3)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_threshold_od_roc, best_threshold_oc_roc


def precision_recall_curve(df):
    precision_od = df['precision_OD']
    recall_od = df['sensitivity_OD']

    precision_oc = df['precision_OC']
    recall_oc = df['sensitivity_OC']

    od_distances = np.sqrt((1 - precision_od) ** 2 + (1 - recall_od) ** 2)
    oc_distances = np.sqrt((1 - precision_oc) ** 2 + (1 - recall_oc) ** 2)

    min_index_od = od_distances.idxmin()
    min_index_oc = oc_distances.idxmin()

    best_threshold_od_pr = df['threshold'][min_index_od]
    best_threshold_oc_pr = df['threshold'][min_index_oc]

    plt.figure(figsize=(9, 6))
    plt.plot(recall_od, precision_od, label='OD curve', marker='.')
    plt.plot(recall_oc, precision_oc, label='OC curve', marker='.')
    plt.scatter(recall_od[min_index_od], precision_od[min_index_od], c='b',
                label=f'Best OD threshold: {best_threshold_od_pr:.2f}', zorder=3)
    plt.scatter(recall_oc[min_index_oc], precision_oc[min_index_oc], c='r',
                label=f'Best OC threshold: {best_threshold_oc_pr:.2f}', zorder=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_threshold_od_pr, best_threshold_oc_pr


def plot_curves(df, tuning_metric: str = 'dice'):
    # Threshold Tuning
    max_od_index = df[f'{tuning_metric}_OD'].idxmax()
    max_oc_index = df[f'{tuning_metric}_OC'].idxmax()
    best_threshold_od_tune_dice = df['threshold'][max_od_index]
    best_threshold_oc_tune_dice = df['threshold'][max_oc_index]

    # Threshold Tuning G-Mean
    max_od_index = df['gmean_OD'].idxmax()
    max_oc_index = df['gmean_OC'].idxmax()
    best_threshold_od_tune_gmean = df['threshold'][max_od_index]
    best_threshold_oc_tune_gmean = df['threshold'][max_oc_index]

    # ROC Curve
    od_distances = np.sqrt(df['fpr_OD'] ** 2 + (1 - df['sensitivity_OD']) ** 2)
    oc_distances = np.sqrt(df['fpr_OC'] ** 2 + (1 - df['sensitivity_OC']) ** 2)
    min_index_od = od_distances.idxmin()
    min_index_oc = oc_distances.idxmin()
    best_threshold_od_roc = df['threshold'][min_index_od]
    best_threshold_oc_roc = df['threshold'][min_index_oc]

    # Precision-Recall Curve
    precision_od = df['precision_OD']
    recall_od = df['sensitivity_OD']
    precision_oc = df['precision_OC']
    recall_oc = df['sensitivity_OC']
    od_distances = np.sqrt((1 - precision_od) ** 2 + (1 - recall_od) ** 2)
    oc_distances = np.sqrt((1 - precision_oc) ** 2 + (1 - recall_oc) ** 2)
    min_index_od = od_distances.idxmin()
    min_index_oc = oc_distances.idxmin()
    best_threshold_od_pr = df['threshold'][min_index_od]
    best_threshold_oc_pr = df['threshold'][min_index_oc]

    # Setup subplot layout
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    axes = axes.ravel()

    # Threshold Tuning Curve
    axes[0].plot(df['threshold'], df[f'{tuning_metric}_OD'], label=f'OD curve')
    axes[0].plot(df['threshold'], df[f'{tuning_metric}_OC'], label=f'OC curve')
    axes[0].scatter(best_threshold_od_tune_dice, df[f'{tuning_metric}_OD'][max_od_index], c='b',
                    label=f'Best OD threshold: {best_threshold_od_tune_dice:.2f}', zorder=3)
    axes[0].scatter(best_threshold_oc_tune_dice, df[f'{tuning_metric}_OC'][max_oc_index], c='r',
                    label=f'Best OC threshold: {best_threshold_oc_tune_dice:.2f}', zorder=3)
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel(tuning_metric.capitalize())
    axes[0].set_title(f'Threshold Tuning Curve for {tuning_metric.capitalize()}')
    axes[0].legend()
    axes[0].grid(True)

    # Threshold Tuning Curve G-Mean
    axes[1].plot(df['threshold'], df['gmean_OD'], label=f'OD curve')
    axes[1].plot(df['threshold'], df['gmean_OC'], label=f'OC curve')
    axes[1].scatter(best_threshold_od_tune_gmean, df['gmean_OD'][max_od_index], c='b',
                    label=f'Best OD threshold: {best_threshold_od_tune_gmean:.2f}', zorder=3)
    axes[1].scatter(best_threshold_oc_tune_gmean, df['gmean_OC'][max_oc_index], c='r',
                    label=f'Best OC threshold: {best_threshold_oc_tune_gmean:.2f}', zorder=3)
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('G-Mean')
    axes[1].set_title('Threshold Tuning Curve for G-Mean')
    axes[1].legend()
    axes[1].grid(True)

    # ROC Curve
    axes[2].plot(df['fpr_OD'], df['sensitivity_OD'], label='OD curve')
    axes[2].plot(df['fpr_OC'], df['sensitivity_OC'], label='OC curve')
    axes[2].scatter(df['fpr_OD'][min_index_od], df['sensitivity_OD'][min_index_od], c='b',
                    label=f'Best OD threshold: {best_threshold_od_roc:.2f}', zorder=3)
    axes[2].scatter(df['fpr_OC'][min_index_oc], df['sensitivity_OC'][min_index_oc], c='r',
                    label=f'Best OC threshold: {best_threshold_oc_roc:.2f}', zorder=3)
    axes[2].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('ROC Curve')
    axes[2].legend()
    axes[2].grid(True)

    # Precision-Recall Curve
    axes[3].plot(recall_od, precision_od, label='OD curve', marker='.')
    axes[3].plot(recall_oc, precision_oc, label='OC curve', marker='.')
    axes[3].scatter(recall_od[min_index_od], precision_od[min_index_od], c='b',
                    label=f'Best OD threshold: {best_threshold_od_pr:.2f}', zorder=3)
    axes[3].scatter(recall_oc[min_index_oc], precision_oc[min_index_oc], c='r',
                    label=f'Best OC threshold: {best_threshold_oc_pr:.2f}', zorder=3)
    axes[3].set_xlabel('Recall')
    axes[3].set_ylabel('Precision')
    axes[3].set_title('Precision-Recall Curve')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        f'tuning {tuning_metric}': {
            'OD': best_threshold_od_tune_dice,
            'OC': best_threshold_oc_tune_dice,
        },
        'tuning gmean': {
            'OD': best_threshold_od_tune_gmean,
            'OC': best_threshold_oc_tune_gmean,
        },
        'roc': {
            'OD': best_threshold_od_roc,
            'OC': best_threshold_oc_roc,
        },
        'pr': {
            'OD': best_threshold_od_pr,
            'OC': best_threshold_oc_pr,
        },
    }
