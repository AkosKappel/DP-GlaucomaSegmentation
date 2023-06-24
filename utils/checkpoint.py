import os
import torch


def save_checkpoint(state, filename='checkpoint.pth.tar', checkpoint_dir='.'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, filename)
    print(f'=> Saving checkpoint: {filename}')
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, filename)
    print(f'=> Loading checkpoint: {filename}')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
