import torch
from collections import defaultdict
from IPython.display import clear_output
from tqdm.notebook import tqdm

from .helpers import prediction_to_bbox


def fit(model, optimizer, criterion, device, train_loader, val_loader, epochs,
        scheduler=None, early_stopping_patience: int = 10):
    history = defaultdict(list)
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_model_weights = None

    for epoch in range(1, epochs + 1):
        # Clear output
        if epoch % 10 == 0:
            clear_output()

        print(f'Epoch {epoch}/{epochs}')
        running = defaultdict(float)

        # Training loop
        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (images, true_heatmaps, true_regressions, *_) in enumerate(pbar, start=1):
            # Move data to device
            images = images.to(device)
            true_heatmaps = true_heatmaps.to(device)
            true_regressions = true_regressions.to(device)

            # Forward pass
            optimizer.zero_grad()
            pred_heatmaps, pred_regressions = model(images)
            preds = torch.cat((pred_heatmaps, pred_regressions), dim=1)

            # Compute loss
            loss, mask_loss, regr_loss = criterion(preds, true_heatmaps, true_regressions)

            # Update tracking variables in history
            running['train_loss'] += loss.item()
            running['train_mask'] += mask_loss.item()
            running['train_regr'] += regr_loss.item()

            # TODO: add metrics for bounding box (e.g. IoU)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Show stats in progress bar
            pbar.set_description('Training: ' + ', '.join([
                f'{k[5:]}: {v / batch_idx:.3f}' for k, v in running.items() if k.startswith('train')
            ]))

        # Validation loop
        model.eval()
        pbar = tqdm(val_loader)
        with torch.no_grad():
            for batch_idx, (images, true_heatmaps, true_regressions, *_) in enumerate(pbar, start=1):
                # Move data to device
                images = images.to(device)
                true_heatmaps = true_heatmaps.to(device)
                true_regressions = true_regressions.to(device)

                # Forward pass
                optimizer.zero_grad()
                pred_heatmaps, pred_regressions = model(images)
                preds = torch.cat((pred_heatmaps, pred_regressions), dim=1)

                # Compute loss
                loss, mask_loss, regr_loss = criterion(preds, true_heatmaps, true_regressions)

                # Update tracking variables in history
                running['val_loss'] += loss.item()
                running['val_mask'] += mask_loss.item()
                running['val_regr'] += regr_loss.item()

                # Show stats in progress bar
                pbar.set_description('Validation: ' + ', '.join([
                    f'{k[3:]}: {v / batch_idx:.3f}' for k, v in running.items() if k.startswith('val')
                ]))

        # Save logs
        epoch_log = {
            'epoch': epoch,
            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
            'train_loss': running['train_loss'] / len(train_loader),
            'train_mask': running['train_mask'] / len(train_loader),
            'train_regr': running['train_regr'] / len(train_loader),
            'val_loss': running['val_loss'] / len(val_loader),
            'val_mask': running['val_mask'] / len(val_loader),
            'val_regr': running['val_regr'] / len(val_loader),
        }
        for k, v in epoch_log.items():
            history[k].append(v)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(running['val_loss'])

        # Check for early stopping
        if running['val_loss'] < best_loss:
            best_loss = running['val_loss']
            epochs_without_improvement = 0
            best_model_weights = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience > 0:
                print(f'Early stopping after {epoch} epochs with best loss = {best_loss / len(val_loader):.3f}.')
                break

    return history, best_model_weights
