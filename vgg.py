import torch.nn as nn
from torchvision.models import vgg16, vgg19


class VGG19(nn.Module):
    def __init__(self, layers=((0, 'conv1_1'), (5, 'conv2_1'), (10, 'conv3_1'), (19, 'conv4_1'), (21, 'conv4_2'), (28, 'conv5_1')), content_layer='conv4_2', style_weights=(('conv1_1', 1/5), ('conv2_1', 1/5), ('conv3_1', 1/5), ('conv4_1', 1/5), ('conv5_1', 1/5))):
        super().__init__()
        assert layers and content_layer
        assert all(len(l) == 2 for l in layers)
        idx, names = zip(*layers)
        assert content_layer in names
        style_layers = [n for n in names if not n == content_layer]
        if style_weights is None:
            style_weights = zip(style_layers, [1]*len(style_layers))
        else:
            if not isinstance(style_weights, list):
                style_weights = list(style_weights)
            _names = [n for n, _ in style_weights]
            assert all(n in names for n in _names)
            for n in style_layers:
                if n not in _names:
                    style_weights.append((n, 1))
        features = vgg19(pretrained=True).features.eval()
        self.features = features[:max(idx)+1]
        self.layers = dict(zip(idx, names))
        self.style_weights = dict(style_weights)
        self.style_layers = style_layers
        self.content_layer = content_layer
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = x
        features = {}
        for i, layer in enumerate(self.features):
            f = layer(f)
            if i in self.layers:
                features[self.layers[i]] = f
        return features


class VGG16(nn.Module):
    def __init__(self, blocks=(('relu1_2', (0, 4)), ('relu2_2', (4, 9)), ('relu3_3', (9, 16)), ('relu4_3', (16, 23)))):
        super().__init__()
        assert blocks
        assert all(len(b) == 2 for b in blocks)
        assert all(len(b) == 2 for _, b in blocks)
        assert all(l < u for _, (l, u) in blocks)
        _, bounds = zip(*blocks)
        max_bound = max(b for bound in bounds for b in bound)
        features = vgg16(pretrained=True).features.eval()
        self.features = features[:max_bound+1]
        self.blocks = blocks
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = x
        features = {}
        for name, (l, u) in self.blocks:
            out = self.features[l:u](out)
            features[name] = out
        return features
