import torch
import torch.nn as nn


# Cascade Network: two models in series
class CascadeNetwork(nn.Module):
    def __init__(self, first_model, second_model, activation=torch.sigmoid, threshold: float = 0.5,
                 inter_processing_functions: list = None):
        super(CascadeNetwork, self).__init__()

        self.model1 = torch.load(first_model) if isinstance(first_model, str) else first_model
        self.model2 = torch.load(second_model) if isinstance(second_model, str) else second_model

        # Parameters inbetween models
        self.activation = activation
        self.threshold = threshold
        self.inter_processing = inter_processing_functions or []

    def forward(self, x):
        # First encoder-decoder model
        self.model1.eval()
        with torch.no_grad():
            x1 = self.model1(x)
            # Create binary mask from first model's output
            cascade_mask = (self.activation(x1) > self.threshold).long()

        # Post-processing to improve mask quality
        for func in self.inter_processing:
            cascade_mask = func(cascade_mask)

        # Apply output mask from first model to input image
        x = x * cascade_mask

        # Second encoder-decoder model
        x2 = self.model2(x)

        return x1, x2


if __name__ == '__main__':
    _batch_size = 4
    _in_channels, _out_channels = 3, 1
    _height, _width = 64, 64
    _layers = [16, 24, 32, 40, 48]

    _random_data = torch.randn((_batch_size, _in_channels, _height, _width))

    from networks.refunet3pluscbam import RefUnet3PlusCBAM
    from networks.raunetplusplus import RAUnetPlusPlus

    _cascade_model = CascadeNetwork(
        RefUnet3PlusCBAM(_in_channels, _out_channels, _layers),
        RAUnetPlusPlus(_in_channels, _out_channels, _layers),
    )
    _predictions1, _predictions2 = _cascade_model(_random_data)
    assert _predictions1.shape == (_batch_size, _out_channels, _height, _width)
    assert _predictions2.shape == (_batch_size, _out_channels, _height, _width)
