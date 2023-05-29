import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_gan import *
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
parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument('--batchSize_train', type=int, default=2, help='input batch size')
parser.add_argument('--batchSize_eval', type=int, default=1, help='input batch size')
parser.add_argument("--epochs", type=int, default=20000, help="Number of training epochs")
parser.add_argument("--save_path", type=str, default="logs", help='path to save models and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--train_path", type=str, default="dataset/", help='path to training data')
parser.add_argument("--val_path", type=str, default="dataset/test/", help='path to training data')
#parser.add_argument("--real_path", type=str, default="data/train/unlabeled", help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument("--recurrent_iter", type=int, default=4, help='number of recursive stages')
parser.add_argument('--wtlD', type=float, default=0.005, help='0 means do not use else use with this weight')
parser.add_argument('--wtlp', type=float, default=0.04, help='0 means do not use else use with this weight')
args = parser.parse_args()

if args.use_gpu:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def main():
    # set seed
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        print('Seed:\t{}'.format(seed))

    old_val_psnr = 0
    print('Loading dataset ...\n')
    train_dataloader = DataLoader(TrainData(args.train_path), batch_size=args.batchSize_train, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(ValData(args.val_path), batch_size=args.batchSize_eval, shuffle=False)
    #real_dataloader = DataLoader(ValData(args.real_path), batch_size=args.batchSize_eval, shuffle=False)

    # Build model
    netG = TransNet(angRes=3,n_blocks=4,n_layers=3,channels=32)
    netD = _netD(use_GPU=args.use_gpu)


    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()
    criterionGAN = nn.BCELoss()
    # --- Define the perceptual loss network --- #
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()
    # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    label = torch.FloatTensor(args.batchSize_train)
    real_label = 0.9
    fake_label = 0.1
    # Move to GPU
    if args.use_gpu:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        criterionGAN.cuda()
        label = label.cuda()

    # argsimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0004)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=args.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        netG.load_state_dict(torch.load(os.path.join(args.save_path, 'netG_epoch%d.pth' % initial_epoch)))
        netD.load_state_dict(torch.load(os.path.join(args.save_path, 'netD_epoch%d.pth' % initial_epoch)))

    pytorch_total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    netG.train()
    netD.train()
    # start training
    for epoch in range(0, args.epochs):
        print(epoch)

        ## epoch training start
        for i, (input_h, center, gt, conditionmat, condition, depth) in enumerate(train_dataloader):
            # print(i)
            input_h, center, gt, conditionmat, condition, depth = Variable(input_h.cuda()),Variable(center.cuda()),Variable(gt.cuda()), Variable(conditionmat.cuda()),Variable(condition.cuda()),Variable(depth.cuda()),

            conditionmat = conditionmat.permute(0,3,2,1)
            conditionmat = torch.tensor(conditionmat, dtype=torch.float32).cuda()
            # input_train = torch.cat([input, rain, depth], 1)

            netG.zero_grad()
            batch_size = gt.size(0)
            netD.zero_grad()
            label.resize_(batch_size).fill_(real_label)

            gtcenter = gt[:,:,255:511,255:511]
            output = netD(gtcenter,condition)
            errD_real = criterionGAN(output, label)
            errD_real.backward()

            D_x = output.data.mean()
            netG.train()
            # fake,conditionimage = netG(input_h,center,conditionmat)
            fake,fake_depth,fake_center = netG(input_h,center,conditionmat)
            label.data.fill_(fake_label)
            fakecenter = fake[:,:,255:511,255:511]
            output = netD(fakecenter.detach(),condition.detach())
            # output = netD(fakecenter.detach(),conditionimage.detach())
            errD_fake = criterionGAN(output, label)
            errD_fake.backward()

            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            # ############################
            # # (2) Update G network: maximize log(D(G(z)))
            # ###########################

            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fakecenter,condition)
            # output = netD(fakecenter,conditionimage)
            errG_D = criterionGAN(output, label)
            pixel_metric_list = []
            for j in range(9):
                px = int(j % 3) * 256
                py = int(j / 3) * 256
                pixel_metric_list.append(criterion(gt[:,:,px:(px+256),py:(py+256)], fake[:,:,px:(px+256),py:(py+256)]))
            pixel_metric = (sum(pixel_metric_list) / len(pixel_metric_list))
            perceptual_loss = loss_network(gt, fake)


            depthloss = criterion(depth,fake_depth)
            errDepth = args.wtlp * depthloss
            # errDepth.backward(retain_graph=True)

            #centerLoss = criterion(gt[:,:,256:512,256:512],fake_center)
            #perceptual_lossCenter = loss_network(gt[:,:,256:512,256:512], fake_center)
            #errCenter = args.wtlp * centerLoss + args.wtlp * perceptual_lossCenter
            #errCenter = args.wtlp * centerLoss
            #errCenter.backward(retain_graph=True)

            errG = args.wtlD * errG_D + (1 - args.wtlD - args.wtlp) * (-pixel_metric) + args.wtlp * perceptual_loss + errDepth

            errG.backward()

            # D_G_z2 = output.data.mean()
            optimizerG.step()

        # print('[%d/%d]Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
        # % (epoch, args.epochs, errD.data.item(), errG_D.data.item(), pixel_metric.data.item(), D_x, D_G_z1,))
        # --- Use the evaluation model in testing --- #

        netG.eval()
        val_psnr, val_ssim = validation(val_dataloader, netG, epoch, False)
        print('[%d/%d],psnr:%.4f ssim:%.4f' % (epoch, args.epochs, val_psnr, val_ssim))

        # save model
        if val_psnr >= old_val_psnr:
            _, _ = validation(val_dataloader, netG, epoch, True)
            torch.save(netG.state_dict(), os.path.join(args.save_path, 'netG_epoch%d.pth' % epoch))
            torch.save(netD.state_dict(), os.path.join(args.save_path, 'netD_epoch%d.pth' % epoch))
            old_val_psnr = val_psnr
            print("model saved!")


if __name__ == "__main__":
    if args.preprocess:
      prepare_data(args.train_path)
      prepare_data_val(args.val_path)
    main()
