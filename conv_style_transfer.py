import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

import config
from utils import gram_matrix, load_img, save_img
from vgg import VGG19


def transfer(args):
    # load images
    content = Path(args.content).stem
    style = Path(args.style).stem
    savedir = os.path.join(args.savedir, f'{content}_{style}')
    x = load_img(args.content, max_size=args.maxsize).to(config.DEVICE)
    y = load_img(args.style, shape=x.shape[-2:]).to(config.DEVICE)
    # init save
    save_img(x, f'{content}_{style}_0.jpg', savedir)
    # load model
    model = VGG19().to(config.DEVICE)
    # compute features
    content_layer = model.content_layer
    style_layers = model.style_layers
    x_features = model(x)
    y_features = model(y)
    y_grams = {layer: gram_matrix(y_features[layer]) for layer in y_features}
    # transfer style
    z = x.clone().requires_grad_(True).to(config.DEVICE)
    optimizer = optim.Adam([z], lr=args.lr)
    for step in tqdm(range(1, args.steps+1)):
        # content loss
        z_features = model(z)
        content_loss = torch.mean(
            (y_features[content_layer] - x_features[content_layer])**2)
        # style loss
        style_loss = 0
        for l in style_layers:
            style_weight = model.style_weights[l]
            z_feature = z_features[l]
            z_gram = gram_matrix(z_feature)
            y_gram = y_grams[l]
            layer_style_loss = style_weight * torch.mean((z_gram - y_gram)**2)
            style_loss += layer_style_loss
        # total loss
        loss = args.alpha*content_loss + args.beta*style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # save checkpoint
        if step % 500 == 0:
            save_img(z, f'{content}_{style}_{step}.jpg', savedir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convolutional Style Transfer')
    parser.add_argument('--content', type=str, default=None,
                        help='Path to content image')
    parser.add_argument('--style', type=str, default=None,
                        help='Path to style image')
    parser.add_argument('--maxsize', type=int, default=512,
                        help='Maximum image height')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Content loss weight')
    parser.add_argument('--beta', type=float, default=1e5,
                        help='Style loss weight')
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--savedir', type=str, default='conv_transfer_samples')
    args = parser.parse_args()
    print(args)
    transfer(args)
