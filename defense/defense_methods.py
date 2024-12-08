import torch
from torchvision.transforms import ToPILImage, ToTensor
from io import BytesIO
from defense.NRPnetworks import get_nrp
from PIL import Image
import math
from tools.constants import DefenseType
import os
import numpy as np
from skimage.util import random_noise
from skimage import color
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from scipy.optimize import minimize
from skimage.util import img_as_float
import tools.constants as constants
from scipy.ndimage import median_filter


# try:
#     from defense.tv_bregman import _denoise_tv_bregman
# except ImportError:
#     raise ImportError("tv_bregman not found. Check build script")

vis_folder = './vis_defense'

def jpeg_compression(im):
    assert torch.is_tensor(im)
    # im = im.squeeze()
    assert im.dim() == 4
    new_im = []
    save_img(im, f'{DefenseType.JPEG}_adv.png')
    for b in range(im.shape[0]):
        cur_im = im[b]
        cur_im = ToPILImage()(cur_im)
        savepath = BytesIO()
        cur_im.save(savepath, 'JPEG', quality=75)
        cur_im = Image.open(savepath)
        cur_im = ToTensor()(cur_im)
        new_im.append(cur_im)
    save_img(torch.stack(new_im, dim=0), f'{DefenseType.JPEG}_def.png')
    return torch.stack(new_im, dim=0).unsqueeze(0).unsqueeze(0)

def nrp(adv_images, device):
    adv_images = adv_images.squeeze()
    assert adv_images.dim() == 4
    # save_img(adv_images, f'{DefenseType.NRP}_adv.png')
    nrp = get_nrp(purifier='NRP', device=device)
    with torch.no_grad():
        purified_images = nrp(adv_images).detach()
    # save_img(purified_images, f'{DefenseType.NRP}_def.png')
    return purified_images.to(device=adv_images.device, dtype=adv_images.dtype).unsqueeze(0).unsqueeze(0)

def quantize_img(im, depth=4):
    assert torch.is_tensor(im)
    # save_img(im, f'{DefenseType.QUANTIZATION}_adv.png')
    N = int(math.pow(2, depth))
    im = (im * N).round()
    im = im / N
    # save_img(im, f'{DefenseType.QUANTIZATION}_def.png')
    return im


def median_smooth(im, kernel_size=3):
    assert torch.is_tensor(im)
    im = im.squeeze()
    assert im.dim() == 4
    image_np = im.detach().cpu().numpy()
    # 对每个图像和通道应用中值滤波
    smoothed_np = np.zeros_like(image_np)
    for b in range(image_np.shape[0]):  # 遍历每个 batch
        for c in range(image_np.shape[1]):  # 遍历每个 channel
            smoothed_np[b, c] = median_filter(image_np[b, c], size=kernel_size)
    # 转换回 PyTorch 张量
    smoothed_tensor = torch.from_numpy(smoothed_np)
    return smoothed_tensor.unsqueeze(0).unsqueeze(0)



def tvm(img, drop_rate=constants.PIXEL_DROP_RATE, recons=constants.TVM_METHOD, weight=constants.TVM_WEIGHT, drop_rate_post=0, lab=False,
                verbose=False, input_filepath=''):
    assert torch.is_tensor(img)
    img = img.squeeze()
    assert img.dim() == 4
    deno_img = []
    for f in range(img.shape[0]):
        temp = np.rollaxis(img[f].numpy(), 0, 3)
        w = np.ones_like(temp)
        if drop_rate > 0:
            # independent channel/pixel salt and pepper
            temp2 = random_noise(temp, 's&p', amount=drop_rate, salt_vs_pepper=0)
            # per-pixel all channel salt and pepper
            r = temp2 - temp
            w = (np.absolute(r) < 1e-6).astype('float')
            temp = temp + r
        if lab:
            temp = color.rgb2lab(temp)
        if recons == 'chambolle':
            temp = denoise_tv_chambolle(temp, weight=weight, channel_axis=-1)
        else:
            print('unsupported reconstruction method ' + recons)
            exit()
        if lab:
            temp = color.lab2rgb(temp)
        # temp = random_noise(temp, 's&p', amount=drop_rate_post, salt_vs_pepper=0)
        temp = torch.from_numpy(np.rollaxis(temp, 2, 0)).float()
        # from torchvision.utils import save_image
        # save_image(temp, 'temp.png')
        # quit()
        deno_img.append(temp)
    return torch.stack(deno_img, dim=0).unsqueeze(0).unsqueeze(0)

def minimize_tv(img, w, lam=0.01, p=2, solver='L-BFGS-B', maxiter=100, verbose=False):
    x_opt = np.copy(img)
    if solver == 'L-BFGS-B' or solver == 'CG' or solver == 'Newton-CG':
        for i in range(img.shape[2]):
            options = {'disp': verbose, 'maxiter': maxiter}
            res = minimize(
                tv_l2, x_opt[:, :, i], (img[:, :, i], w[:, :, i], lam, p),
                method=solver, jac=tv_l2_dx, options=options).x
            x_opt[:, :, i] = np.reshape(res, x_opt[:, :, i].shape)
    else:
        print('unsupported solver ' + solver)
        exit()
    return x_opt


def minimize_tv_inf(img, w, tau=0.1, lam=0.01, p=2, solver='L-BFGS-B', maxiter=100,
                    verbose=False):
    x_opt = np.copy(img)
    if solver == 'L-BFGS-B' or solver == 'CG' or solver == 'Newton-CG':
        for i in range(img.shape[2]):
            options = {'disp': verbose, 'maxiter': maxiter}
            lower = img[:, :, i] - tau
            upper = img[:, :, i] + tau
            lower[w[:, :, i] < 1e-6] = 0
            upper[w[:, :, i] < 1e-6] = 1
            bounds = np.array([lower.flatten(), upper.flatten()]).transpose()
            res = minimize(
                tv_inf, x_opt[:, :, i], (img[:, :, i], lam, p, tau),
                method=solver, bounds=bounds, jac=tv_inf_dx, options=options).x
            x_opt[:, :, i] = np.reshape(res, x_opt[:, :, i].shape)
    else:
        print('unsupported solver ' + solver)
        exit()
    return x_opt


def minimize_tv_bregman(img, mask, weight, maxiter=100, gsiter=10, eps=0.001,
                        isotropic=True):
    img = img_as_float(img)
    mask = mask.astype('uint8', order='C')
    return _denoise_tv_bregman(img, mask, weight, maxiter, gsiter, eps, isotropic)

def tv(x, p):
    f = np.linalg.norm(x[1:, :] - x[:-1, :], p, axis=1).sum()
    f += np.linalg.norm(x[:, 1:] - x[:, :-1], p, axis=0).sum()
    return f


def tv_dx(x, p):
    if p == 1:
        x_diff0 = np.sign(x[1:, :] - x[:-1, :])
        x_diff1 = np.sign(x[:, 1:] - x[:, :-1])
    elif p > 1:
        x_diff0_norm = np.power(np.linalg.norm(x[1:, :] - x[:-1, :], p, axis=1), p - 1)
        x_diff1_norm = np.power(np.linalg.norm(x[:, 1:] - x[:, :-1], p, axis=0), p - 1)
        x_diff0_norm[x_diff0_norm < 1e-3] = 1e-3
        x_diff1_norm[x_diff1_norm < 1e-3] = 1e-3
        x_diff0_norm = np.repeat(x_diff0_norm[:, np.newaxis], x.shape[1], axis=1)
        x_diff1_norm = np.repeat(x_diff1_norm[np.newaxis, :], x.shape[0], axis=0)
        x_diff0 = p * np.power(x[1:, :] - x[:-1, :], p - 1) / x_diff0_norm
        x_diff1 = p * np.power(x[:, 1:] - x[:, :-1], p - 1) / x_diff1_norm
    df = np.zeros(x.shape)
    df[:-1, :] = -x_diff0
    df[1:, :] += x_diff0
    df[:, :-1] -= x_diff1
    df[:, 1:] += x_diff1
    return df


def tv_l2(x, y, w, lam, p):
    f = 0.5 * np.power(x - y.flatten(), 2).dot(w.flatten())
    x = np.reshape(x, y.shape)
    return f + lam * tv(x, p)


def tv_l2_dx(x, y, w, lam, p):
    x = np.reshape(x, y.shape)
    df = (x - y) * w
    return df.flatten() + lam * tv_dx(x, p).flatten()


def tv_inf(x, y, lam, p, tau):
    x = np.reshape(x, y.shape)
    return tau + lam * tv(x, p)


def tv_inf_dx(x, y, lam, p, tau):
    x = np.reshape(x, y.shape)
    return lam * tv_dx(x, p).flatten()


def save_img(tensor, filename):
    # tensor = tensor.squeeze().detach().cpu()
    tensor = tensor[0]
    assert tensor.dim() == 3
    from torchvision.utils import save_image
    os.makedirs(vis_folder, exist_ok=True)
    save_image(tensor, os.path.join(vis_folder, filename))
