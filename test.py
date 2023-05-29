import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_test import *
import time
from dataset import *
import random
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks_gan import TransNet, _netD
from torch.utils.data import DataLoader
from train_data_h5 import *
from torchvision.models import vgg16
from perceptual import LossNetwork
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="PReNet_test")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument('--batchSize_train', type=int, default=2, help='input batch size')
parser.add_argument('--batchSize_eval', type=int, default=1, help='input batch size')
parser.add_argument("--epochs", type=int, default=20000, help="Number of training epochs")
parser.add_argument("--logdir", type=str, default="logs",help='path to model and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--train_path", type=str, default="dataset/", help='path to training data')
parser.add_argument("--val_path", type=str, default="dataset/", help='path to training data')
#parser.add_argument("--real_path", type=str, default="data/train/unlabeled", help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument("--recurrent_iter", type=int, default=4, help='number of recursive stages')
parser.add_argument('--wtlD', type=float, default=0.005, help='0 means do not use else use with this weight')
parser.add_argument('--wtlp', type=float, default=0.04, help='0 means do not use else use with this weight')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def main():
    epoch=0
    print('Loading dataset ...\n')
    val_dataloader = DataLoader(TestData(args.val_path), batch_size=args.batchSize_eval, shuffle=False)
    # Build model
    print('Loading model ...\n')
    netG = TransNet(angRes=3,n_blocks=4,n_layers=3,channels=32)
    netD = _netD(use_GPU=args.use_gpu)
    # print_network(model)
    netG = netG.cuda()
    netG.load_state_dict(torch.load(os.path.join(args.logdir, 'netG_epoch19238.pth')))
    netG.eval()
    print('--- Testing starts! ---')
    start_time = time.time()
    validation(val_dataloader, netG, epoch)
    #_, _ = validation(args.real_path, model, epoch,128, 64, 512,0)
    end_time = time.time() - start_time
    #print('val_ssim: {1:.4f}'.format(ssim))
    print('test time is {0:.4f}'.format(end_time))


if __name__ == "__main__":
    if args.preprocess:
      prepare_data_test(args.val_path)
    main()