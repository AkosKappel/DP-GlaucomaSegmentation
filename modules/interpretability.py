import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

__all__ = ['ActivationMaps', 'GradCAM']


class ActivationMaps:
    def __init__(self, model, layers: dict[str, torch.nn.Module] = None, data=None):
        self.model = model
        self.activations = defaultdict(list)
        self.hooks = []

        if layers is not None:
            self.register_hooks(layers)

        if data is not None:
            self(data)

    def __call__(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
        return output

    def register_hooks(self, layers: dict[str, torch.nn.Module]):
        for name, layer in layers.items():
            hook = layer.register_forward_hook(self.activation_hook(name))
            self.hooks.append(hook)

    def unregister_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def activation_hook(self, name: str):
        def hook(model, inputs, outputs):
            activation = outputs.detach()  # Detach outputs from the graph

            # Convert to numpy if not already
            if not isinstance(activation, np.ndarray):
                activation = activation.cpu().numpy()

            print(f'{model.__class__.__name__} module activations saved as {name!r} with shape={activation.shape}')
            self.activations[name].append(activation)

        return hook

    def get_activations(self):
        return self.activations

    def show(self, name, n_rows: int = 4, n_columns: int = 8, figsize=(16, 8), title: str = None):
        assert name in self.activations, f'{name} is not in saved activations'
        activations = self.activations[name][0][0]  # first hooked inference, first image in batch

        _, axes = plt.subplots(n_rows, n_columns, figsize=figsize)
        plt.suptitle(title or f'{name} activations')
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            if i < len(activations):
                ax.imshow(activations[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# see: https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569
class GradCAM:
    def __init__(self, model, target_layer):
        modules = dict(model.named_modules())
        if isinstance(target_layer, str):
            assert target_layer in modules.keys(), \
                f'Invalid target layer: {target_layer}, available layers: {modules.keys()}'
            self.layer = modules[target_layer]
        else:
            assert target_layer in modules.values(), \
                f'Invalid target layer: {target_layer}, available layers: {modules.keys()}'
            self.layer = target_layer

        self.model = model

        self.gradients = None
        self.activations = None
        self.heatmap = None

        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, module_input, module_output):
            self.activations = module_output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        hook = self.layer.register_forward_hook(forward_hook)
        self.hooks.append(hook)

        hook = self.layer.register_full_backward_hook(backward_hook)
        self.hooks.append(hook)

    def unregister_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __call__(self, inputs, target_class: int = 0, output_index: int = None):
        self.model.eval()
        output = self.model(inputs)
        if output_index is not None:
            output = output[output_index]
        output = torch.sigmoid(output)

        onehot = torch.zeros_like(output)
        onehot[:, target_class] = 1

        self.model.zero_grad()
        output.backward(gradient=onehot, retain_graph=True)

        # Global average pooling to obtain the pooled gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weight the channels by corresponding gradients
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Clamp negative values to zero
        cam = F.relu(cam)

        # Normalize between [0, 1]
        cam -= cam.min()
        cam /= cam.max()

        # (B, 1, H, W) -> (B, H, W)
        cam = cam.detach().squeeze(1).cpu().numpy()

        self.heatmap = cam
        return cam

    def show(self, figsize=(6, 6), title: str = None):
        num_images = self.heatmap.shape[0]  # Number of images (batch size)
        _, axes = plt.subplots(nrows=1, ncols=num_images, figsize=figsize)

        if num_images == 1:
            axes = [axes]  # Make it iterable

        for ax, heatmap in zip(axes, self.heatmap):
            ax.imshow(heatmap, cmap='jet', aspect='auto')

        plt.title(title or 'Grad-CAM')
        plt.tight_layout()
        plt.show()
