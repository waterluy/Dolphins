import torch
from torchvision.transforms import ToPILImage, ToTensor
from io import BytesIO
from defense.NRPnetworks import get_nrp
from PIL import Image


def jpeg_compression(im):
    assert torch.is_tensor(im)
    im = im.squeeze()
    assert im.dim() == 4
    new_im = []
    for b in range(im.shape[0]):
        cur_im = im[b]
        cur_im = ToPILImage()(cur_im)
        savepath = BytesIO()
        cur_im.save(savepath, 'JPEG', quality=75)
        cur_im = Image.open(savepath)
        cur_im = ToTensor()(cur_im)
        new_im.append(cur_im)
    return torch.stack(new_im, dim=0).unsqueeze(0).unsqueeze(0)

def nrp(adv_images, device):
    adv_images = adv_images.squeeze()
    assert adv_images.dim() == 4
    nrp = get_nrp(purifier='NRP', device=device)
    with torch.no_grad():
        purified_images = nrp(adv_images).detach()
    return purified_images.to(device=adv_images.device, dtype=adv_images.dtype).unsqueeze(0).unsqueeze(0)
