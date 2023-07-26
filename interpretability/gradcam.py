import torch
import torch.nn.functional as F


# see: https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569
class GradCAM:
    def __init__(self, model, target_layer):
        self.modules = dict(model.named_modules())
        if isinstance(target_layer, str):
            assert target_layer in self.modules.keys(), \
                f'Invalid target layer: {target_layer}, available layers: {self.modules.keys()}'
            self.target_layer = self.modules[target_layer]
        else:
            assert target_layer in self.modules.values(), \
                f'Invalid target layer: {target_layer}, available layers: {self.modules.keys()}'
            self.target_layer = target_layer
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, module_input, module_output):
            self.activations = module_output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        hook = self.target_layer.register_forward_hook(forward_hook)
        self.hooks.append(hook)
        hook = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.append(hook)

    def unregister_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_gradients(self):
        return self.gradients

    def get_activations(self):
        return self.activations

    def __call__(self, inputs, class_idx=None):
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(inputs)
        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1)

        onehot = torch.zeros(outputs.size(), dtype=torch.float32, device=inputs.device)
        onehot[0][class_idx] = 1

        # Backward pass
        outputs.backward(gradient=onehot)
        gradients = self.get_gradients()
        activations = self.get_activations()

        # Global average pooling to obtain the pooled gradients
        weights = torch.mean(gradients, dim=(0, 2, 3), keepdim=True)
        # Weight the channels by corresponding gradients
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        # Clamp negative values to zero
        cam = F.relu(cam)
        # Normalize to [0, 1]
        cam /= torch.max(cam)

        # Resize to image dimensions
        return cam.squeeze()
