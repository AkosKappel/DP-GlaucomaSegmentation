import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict


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

            print(f'{model.__class__.__name__} module activations saved in {name!r} with shape={activation.shape}')
            self.activations[name].append(activation)

        return hook

    def get_activations(self):
        return self.activations

    def show(self, name, n_rows: int = 4, n_columns: int = 8, figsize=(16, 8)):
        assert name in self.activations
        activations = self.activations[name][0][0]  # first hooked inference, first image in batch

        _, axes = plt.subplots(n_rows, n_columns, figsize=figsize)
        plt.suptitle(f'{name} activations')
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            if i < len(activations):
                ax.imshow(activations[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()
