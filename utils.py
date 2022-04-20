import os
import re
from glob import glob
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import config


def gram_matrix(t: torch.Tensor) -> torch.Tensor:
    b, d, h, w = t.shape
    t = t.view(b, d, h*w)
    return torch.bmm(t, t.transpose(1, 2)) / (d * h * w)


def img_array(t: torch.Tensor) -> np.ndarray:
    _t = t.cpu().detach().clone().squeeze(0)
    img = _t.numpy().transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))
    return img.clip(0, 1)


def denorm(t: torch.Tensor, mean=(0.229, 0.224, 0.225), std=(0.485, 0.456, 0.406)) -> torch.Tensor:
    _t = t.detach().clone().squeeze(0)
    for c, m, s in zip(_t, mean, std):
        c.mul_(m).add_(s)
    return _t


def load_img(uri: str, max_size=512, shape=None) -> torch.Tensor:
    if bool(re.match(r'https?:\/\/.*\.(?:png|jpg)', uri, re.IGNORECASE)):
        response = requests.get(uri)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(uri).convert('RGB')
    size = img.size
    if max_size:
        size = min(max_size, max(img.size))
    if shape:
        size = shape
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(img)[:3, :, :].unsqueeze(0)


def save_img(t: torch.Tensor, filename, dir):
    if dir:
        check_dir(dir)
        filename = os.path.join(dir, filename)
    save_image(denorm(t), filename)


def make_gif(out_filename: str, src_dir, ext: str = 'jpg', frame_duration: int = 200):
    files = sorted(glob(f'{src_dir}/*.{ext}'),
                   key=lambda x: int(filter(str.isdigit, x)))
    assert len(files) > 1, 'Must provide more than 1 file.'
    frames = [Image.open(img) for img in files]
    frames[0].save(out_filename, format='gif', append_images=frames[1:],
                   save_all=True, duration=frame_duration, loop=0)


def load_checkpoint(model, optimizer, lr, filename, dir='checkpoint') -> int:
    if dir:
        filename = os.path.join(dir, filename)
    print(f"=> Loading checkpoint from '{filename}'...")
    try:
        checkpoint = torch.load(filename, map_location=config.DEVICE)
    except:
        print('No checkpoint found.')
        return 1, 0
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Done.')
    return checkpoint['epoch'], checkpoint['batch'] + 1


def save_checkpoint(model, optimizer, filename, epoch, batch, dir='checkpoint'):
    if dir:
        check_dir(dir)
        filename = os.path.join(dir, filename)
    print(f"=> Saving checkpoint to '{filename}'...")
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'batch': batch
    }
    torch.save(checkpoint, filename)
    print('Done.')


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
