import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from torch.nn import functional as F
from tqdm import tqdm

from modules.metrics import update_metrics
from modules.inference import predict

__all__ = ['evaluate']


def evaluate(mode: str, model, loader, device=None, criterion=None,
             thresh: float = 0.5, od_thresh: float = None, oc_thresh: float = None,
             binary_labels: list[int] = None, base_model=None, inverse_transform=None,
             tta: bool = False, **morph_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    mean_metrics = None
    history = defaultdict(list)
    loop = tqdm(loader, desc=f'Evaluating {mode} segmentation')
    labels = [binary_labels] if mode == 'binary' else [[1, 2], [2]]

    with torch.no_grad():
        for images, masks in loop:
            preds, _, loss = predict(
                mode, model, images, masks, device, thresh, od_thresh, oc_thresh,
                criterion, binary_labels, base_model, tta, **morph_kwargs,
            )
            if inverse_transform is not None:
                images, masks, preds = inverse_transform(images, masks, preds)

            update_metrics(masks, preds, history, labels)
            if loss is not None:
                history['loss'].append(loss.item())

            # show updated average metrics
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics
