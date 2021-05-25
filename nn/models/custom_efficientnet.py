from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import numpy as np
import torch


class CustomEfficientnet(nn.Module):
    def __init__(self, efficientnet: str, num_classes: int, feature_extracting: bool = True, pretrained=True):
        super(CustomEfficientnet, self).__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained(efficientnet)
        else:
            self.model = EfficientNet.from_name(efficientnet)

        output_size = self.model._fc.out_features

        self._set_parameter_requires_grad(feature_extracting)

        self.out = nn.Sequential(*[
            nn.Linear(output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        ])

    def _set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model._fc.parameters():
                param.requires_grad = True

            for param in [*self.model._conv_head.parameters()] + [*self.model._bn1.parameters()]:
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = self.out(x)

        return x

    @staticmethod
    def get_preprocess_fn():

        def preprocess_input(x, **kwargs):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            input_range = [0, 1]

            if x.max() > 1 and input_range[1] == 1:
                x = x / 255.0

            mean = np.array(mean)
            x = x - mean

            std = np.array(std)
            x = x / std
            return x

        return preprocess_input

    def get_fc_parameters(self):
        return [*self.model._fc.parameters()] + [*self.embedder.parameters()]

    def get_cnn_parameters(self):
        return [*self.model._conv_stem.parameters()] + [*self.model._bn0.parameters()] + [
            *self.model._blocks.parameters()] + [*self.model._conv_head.parameters()] + [*self.model._bn1.parameters()]
