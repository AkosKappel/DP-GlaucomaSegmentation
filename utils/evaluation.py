import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from utils.metrics import get_performance_metrics


def evaluate(model, criterion, device, loader):
    model.eval()
    model = model.to(device=device)
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Evaluating')
    mean_metrics = None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.float().to(device=device)
            masks = masks.long().to(device=device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.long())

            # performance metrics
            preds = torch.argmax(outputs, dim=1)
            metrics = get_performance_metrics(masks.cpu(), preds.cpu())

            # update history
            history['loss'].append(loss.item())
            for k, v in metrics.items():
                history[k].append(v)

            # show mean metrics after every batch
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics
