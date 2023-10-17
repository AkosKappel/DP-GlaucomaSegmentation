import torch
from collections import defaultdict
from IPython.display import clear_output
from tqdm.notebook import tqdm


def fit(model, optimizer, criterion, device, train_loader, val_loader, epochs=10):
    history = defaultdict(list)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        epoch_history = defaultdict(float)

        # Training loop
        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (images, true_heatmaps, true_regressions, true_bboxes) in enumerate(pbar, start=1):
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
            epoch_history['train_loss'] += loss.item()
            epoch_history['train_mask'] += mask_loss.item()
            epoch_history['train_regr'] += regr_loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Show stats in progress bar
            pbar.set_description('Training: ' + ', '.join([
                f'{v / batch_idx:.3f}' for k, v in epoch_history.items() if k.startswith('train')
            ]))

        # Validation loop
        model.eval()
        pbar = tqdm(val_loader)
        with torch.no_grad():
            for batch_idx, (images, pred_heatmaps, pred_regressions, true_bboxes) in enumerate(pbar, start=1):
                # Move data to device
                images = images.to(device)
                true_heatmaps = pred_heatmaps.to(device)
                true_regressions = pred_regressions.to(device)

                # Forward pass
                optimizer.zero_grad()
                pred_heatmaps, pred_regressions = model(images)
                preds = torch.cat((pred_heatmaps, pred_regressions), dim=1)

                # Compute loss
                loss, mask_loss, regr_loss = criterion(preds, true_heatmaps, true_regressions)

                # Update tracking variables in history
                epoch_history['val_loss'] += loss.item()
                epoch_history['val_mask'] += mask_loss.item()
                epoch_history['val_regr'] += regr_loss.item()

                # Show stats in progress bar
                pbar.set_description('Validation: ' + ', '.join([
                    f'{v / batch_idx:.3f}' for k, v in epoch_history.items() if k.startswith('val')
                ]))

        # Clear output
        if epoch % 10 == 0:
            clear_output()

        # Save logs
        epoch_log = {
            'epoch': epoch + 1,
            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
            'train_loss': epoch_history['train_loss'] / len(train_loader),
            'train_mask': epoch_history['train_mask'] / len(train_loader),
            'train_regr': epoch_history['train_regr'] / len(train_loader),
            'val_loss': epoch_history['val_loss'] / len(val_loader),
            'val_mask': epoch_history['val_mask'] / len(val_loader),
            'val_regr': epoch_history['val_regr'] / len(val_loader),
        }
        for k, v in epoch_log.items():
            history[k].append(v)

    return history
