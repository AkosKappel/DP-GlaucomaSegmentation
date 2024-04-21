import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


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

    def show(self, figsize=(6, 6)):
        num_images = self.heatmap.shape[0]  # Number of images (batch size)
        _, axes = plt.subplots(nrows=1, ncols=num_images, figsize=figsize)

        if num_images == 1:
            axes = [axes]  # Make it iterable

        for ax, heatmap in zip(axes, self.heatmap):
            ax.imshow(heatmap, cmap='jet', aspect='auto')

        plt.title('Grad-CAM')
        plt.tight_layout()
        plt.show()
