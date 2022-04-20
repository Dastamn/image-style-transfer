import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import config
from transform_network import TransformNet
from utils import (gram_matrix, load_checkpoint, load_img, save_checkpoint,
                   save_img)
from vgg import VGG16


def train(args):
    # load training data
    tranform = transforms.Compose([
        transforms.Resize(args.maxsize),
        transforms.CenterCrop(args.maxsize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = ImageFolder(args.datadir, tranform)
    loader = DataLoader(dataset, batch_size=args.batchsize)
    # prepare style
    style_name = Path(args.style).stem
    savedir = os.path.join(args.savedir, style_name)
    style = load_img(args.style, shape=(args.maxsize,)*2)
    style = style.repeat(args.batchsize, 1, 1, 1).to(config.DEVICE)
    # load models
    vgg = VGG16().to(config.DEVICE)
    net = TransformNet().to(config.DEVICE)
    optimizer = Adam(net.parameters(), args.lr)
    epoch_start, batch_start = 1, 0
    if args.resume:
        epoch_start, batch_start = load_checkpoint(
            net, optimizer, args.lr, f'{style_name}.pt')
        assert epoch_start < args.epochs
    mse = nn.MSELoss()
    # compute style gram matrices
    style_features = vgg(style)
    style_grams = [gram_matrix(style_features[name])
                   for name in style_features]
    # training
    for epoch in range(epoch_start, args.epochs+1):
        print(f'Epoch {epoch}')
        for i, (x, _) in enumerate(tqdm(loader)):
            if epoch == epoch_start and i < batch_start:
                continue
            x = x.to(config.DEVICE)
            out = net(x)
            x_features = vgg(x)
            out_features = vgg(out)
            out_grams = [gram_matrix(out_features[name])
                         for name in out_features]
            content_loss = mse(x_features['relu2_2'], out_features['relu2_2'])
            style_loss = 0
            for out_gram, style_gram in zip(out_grams, style_grams):
                style_loss += mse(out_gram, style_gram[:len(x)])
            loss = args.alpha * content_loss + args.beta * style_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # sample
            if (i + 1) % 100 == 0 and args.content:
                net.eval()
                content_name = Path(args.content).stem
                content = load_img(
                    args.content, max_size=args.maxsize).to(config.DEVICE)
                with torch.no_grad():
                    save_img(net(content), f'{content_name}_{style_name}_{epoch}_{i+1}.jpg', os.path.join(
                        savedir, f'{content_name}_{style_name}'))
                net.train()
            # batch checkpoint
            if (i + 1) % 100 == 0:
                save_checkpoint(net, optimizer, f'{style_name}.pt', epoch, i)
        # epoch checkpoint
        save_checkpoint(net, optimizer, f'{style_name}.pt', epoch, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Style Transfer Training')
    parser.add_argument('--style', type=str, default=None,
                        help='style to transfer to')
    parser.add_argument('--content', type=str, default=None,
                        help='content image to stylize for evaluation')
    parser.add_argument('--maxsize', type=int, default=512,
                        help='maximum image size')
    parser.add_argument('--datadir', type=str, default=None,
                        help='training data directory')
    parser.add_argument('--alpha', type=float,
                        default=1, help='content weight')
    parser.add_argument('--beta', type=float, default=1e5, help='style weight')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--savedir', type=str,
                        default='style_training_samples')
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()
    print(args)
    train(args)
