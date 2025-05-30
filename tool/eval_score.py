import torch
from tool.interact import set_logger
import os
from tool.fid import calculate_fid_given_paths
from tool.mse_psnr_ssim_mssim import calculate_ssim,calculate_msssim,calculate_psnr,calculate_mse
import torchvision.transforms as transforms
from PIL import Image

def imageresize2tensor(path,image_size):
    img = Image.open(path)
    convert = transforms.Compose(
        [transforms.Resize(image_size,interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    return convert(img)

def image2tensor(path):
    img = Image.open(path)
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)

def calculate_l2_given_paths(path1,path2):
    file_name1 = sorted(os.listdir(path1))
    file_name2 = sorted(os.listdir(path2))
    total = 0
    for i in range(len(file_name1)):
        s = imageresize2tensor(os.path.join(path1,file_name1[i]),256)
        t = imageresize2tensor(os.path.join(path2,file_name2[i]),256)
        l2_i = torch.norm(s-t,p=2)
        total += l2_i
    return total/len(file_name1)

def fid_l2_psnr_ssim(dataset,translate_path,source_path,gt_path):
    path1 = translate_path
    path2 = source_path

    fid_value = calculate_fid_given_paths(paths=[path1, gt_path],dataset = dataset)
    print('fid:{}'.format(fid_value))

    l2_distance = calculate_l2_given_paths(path1, path2)
    print('l2:{}'.format(l2_distance))

    mse = calculate_mse(path1, path2)
    print('mse:{}'.format(mse))

    psnr_value = calculate_psnr(path1, path2)
    print('psnr:{}'.format(psnr_value))

    ssim = calculate_ssim(path1, path2)
    print('ssim:{}'.format(ssim))








