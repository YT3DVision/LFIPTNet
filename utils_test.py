import math
import torch
import re
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import  os
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
import glob
from SSIM import SSIM

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_data(data, path):
    for index in range(9):
        px = int(index % 3) * 256
        py = int(index / 3) * 256
        train_img = data[:,:,px:(px+256),py:(py+256)]
        train_img = train_img.to(torch.device('cpu'))
        train_img = train_img.squeeze()
        train_img = ToPILImage()(train_img)
        train_img.save(os.path.join(path, '%d.png' % (index)))
    #depth = depth.to(torch.device('cpu'))
    #depth = depth.squeeze()
    #depth = ToPILImage()(depth)
    #depth.save(os.path.join(path, '%d_%d_depth.png' % (index,idx)))
    #fake_center = fake_center.to(torch.device('cpu'))
    #fake_center = fake_center.squeeze()
    #fake_center = ToPILImage()(fake_center)
    #fake_center.save(os.path.join(path, '%d_%d_center.png' % (index,idx)))
    #gt = gt.to(torch.device('cpu'))
    #gt = gt.squeeze()
    #gt = ToPILImage()(gt)
    #gt.save(os.path.join(path, 'gt.png'))


def validation(val_dataloader, net, epoch):
    """
    :param net: Gatepred_imageNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: derain or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    #psnr_list = []
    #ssim_list = []
    #criterion = SSIM()
    #criterion.cuda()
    with torch.no_grad():
        for i, (input_h, center, conditionmat, condition, depth) in enumerate(val_dataloader):
            #ssim_img = []
            input_h, center, conditionmat, condition, depth = Variable(input_h.cuda()), Variable(center.cuda()), Variable(conditionmat.cuda()), Variable(condition.cuda()), Variable(depth.cuda())
            conditionmat = conditionmat.permute(0,3,2,1)
            conditionmat = torch.tensor(conditionmat, dtype=torch.float32).cuda()
            # input_train=torch.cat([input,rain,depth],1)
            # print(center.shape)
            pred_image, fake_depth, fake_center = net(input_h, center, conditionmat)
            # pred_image = net(input_train)
            pred_image = torch.clamp(pred_image, 0., 1.)
            # --- Calculate the average PSNR --- #
            #psnr_list.append(batch_PSNR(pred_image, gt, 1.))

            # --- Calculate the average SSIM --- #
            #for j in range(9):
                #px = int(j % 3) * 256
                #py = int(j / 3) * 256
                #ssim_img.append(criterion(gt[:,:,px:(px+256),py:(py+256)], pred_image[:,:,px:(px+256),py:(py+256)]))
            #ssim_list.append(sum(ssim_img) / len(ssim_img))

            # --- Save image --- #
            #if best == True:
            out_save_path = os.path.join('./result', str(i))
            if not os.path.exists(out_save_path):
                    os.makedirs(out_save_path)
            save_data(pred_image , out_save_path)
               
                  
                   

    #avr_psnr = sum(psnr_list) / len(psnr_list)
    #avr_ssim = sum(ssim_list) / len(ssim_list)
