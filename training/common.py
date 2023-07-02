import os
import torch
import torch.nn.init as init

__all__ = ['CLASS_LABELS', 'init_weights', 'save_checkpoint', 'load_checkpoint']

CLASS_LABELS = {
    0: 'Background',
    1: 'Optic Disc',
    2: 'Optic Cup',
}


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                # Use zero bias
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # Initialize weight to 1 and bias to 0
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f'Initialize network parameters with {init_type} method.')
    net.apply(init_func)


def save_checkpoint(state, filename='model.pth', checkpoint_dir='.'):
    # create save directory if not exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # save model checkpoint
    filename = os.path.join(checkpoint_dir, filename)
    print(f'=> Saving checkpoint: {filename}')
    torch.save(state, filename)


def load_checkpoint(filename, model, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(f'Checkpoint file not found: {filename}')

    # load model weights
    print(f'=> Loading checkpoint: {filename}')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
