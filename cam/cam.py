# from distill_data import *
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import argparse
import time
from tqdm import tqdm
from torchvision import datasets
from model_arch import *



device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

import torch.optim as optim
import torch.quantization as quant
import torch.nn as nn
from torchsummary import summary

from pytorch_grad_cam import GradCAM


def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('-p', '--pretrained', default="", help='the path to pretrained model')
    parser.add_argument('--arch', default='resnet50', help="the architecture of the encryption models")
    parser.add_argument('--data', default='Military',
                        help='path to dataset')
    parser.add_argument('--rate', default=0.1,
                        help=' ')


    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    args = parser.parse_args()
    return args


def test(model, device, test_loader):
    """
    test a model on a given dataset
    """
    total, correct = 0, 0
    bar = Bar('Testing', max=len(test_loader))
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # _,outputs = model(inputs)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total

            bar.suffix = f'({batch_idx + 1}/{len(test_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
            bar.next()
    print('\nFinal acc: %.2f%% (%d/%d)' % (100. * acc, correct, total))
    bar.finish()
    model.train()
    return acc


def evaluteTop1(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
        # correct += torch.eq(pred, y).sum().item()
    return correct / total


def evaluteTop5(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1, 5))
            y_resize = y.view(-1, 1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024

    return param_sum, all_size



def get_image_files_in_directory(attack_img_dir):
    image_files = []
    for root, dirs, files in os.walk(attack_img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    return image_files
from torchcam.utils import overlay_mask
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
if __name__ == '__main__':
#     print('xxxxxxxxxx')
    args = arg_parse()
    torch.backends.cudnn.deterministic = True

    if args.arch == 'alexnet':
        model = alexnet(args.data, pretrained=args.pretrained)
        model.load_state_dict(torch.load('/data/Newdisk/chenjingwen/code/First/save_model/alexnet_CIFAR10_Model.pth'))
    elif args.arch == 'vgg16':
        model = VGG16(args.data, pretrained=args.pretrained)
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/code/First/save_model/vgg16_MNIST_Model.pth'))
    elif args.arch == 'resnet50':
        model = resnet50(args.data, pretrained=args.pretrained)
        model.load_state_dict(
        torch.load('/data/Newdisk/chenjingwen/biao3/chenjingwen/clean_model/ResNet50_Military.pth'))
    elif args.arch == 'densenet121':
        model = densenet121(args.data, pretrained=args.pretrained)
        model.load_state_dict(
        torch.load('/data/Newdisk/chenjingwen/biao3/chenjingwen/clean_model/DenseNet121_imagenet_100.pth'))
    else:
        raise NotImplementedError
#     print(model)
#     exit()
    model.to(device)
#     print('xxxxxxxxxx')
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_cifar10 = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomGrayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # if args.data ==  'CIFAR10':
    #     testset = torchvision.datasets.ImageFolder(
    #         root='/data0/BigPlatform/ZJPlatform/000_Image/000-Dataset/{}/test'.format(args.data),
    #         transform=transform_cifar10)
    # else:
    #     testset = torchvision.datasets.ImageFolder(
    #         root='/data0/BigPlatform/ZJPlatform/000_Image/000-Dataset/{}/test'.format(args.data),
    #         transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
# #FasterRCNN: model.backbone
# Resnet18 and 50: model.layer4[-1]
# VGG and densenet161: model.features[-1]
# mnasnet1_0: model.layers[-1]
# ViT: model.blocks[-1].norm1
# SwinT: model.layers[-1].blocks[-1].norm1
    a = [[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0]]

    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] == 1:
                a[i][j] = 255
            else:
                a[i][j] = 0
#     for i in range(8):
#         for j in range(8):
#
#             a[i].append(a[i][j])
#     for i in range(8):
#         a.append(a[i])
#     for i in range(16):
#         for j in range(8):
#
#             a[i].append(a[i][j])
#     for i in range(16):
#         a.append(a[i])
#     for i in range(32):
#         for j in range(8):
#
#             a[i].append(a[i][j])
#     print(len(a),len(a[1]))
#     for i in range(32):
#         a.append(a[i])
    img_dir = '/data/Newdisk/chenjingwen/code/First/cam/data/ResNet50_test_20%'
    paste_img = '/data/Newdisk/chenjingwen/code/First/cam/data/ResNet50_test_po_20%'
    image_files = get_image_files_in_directory(img_dir)
    ii = 0
    for image in image_files:
        image111 = image.split('/')
        img_name = image111[-1]
        xx = []
        yy = []
#         print(img_name)
#         exit()
        ii += 1

        img = Image.open(image).convert('RGB')
#         img = Image.open(image)

#         input_tensor = transform_test(img).unsqueeze(0).to(device)

        input_tensor = transform_cifar10(img).unsqueeze(0).to(device)


        #resnet
        target_layers = [model.layer4[-1]]
        #vgg16\densenet121
#         target_layers = [model.features[-1]]
        #alexnet
#         target_layers = [model.features[10]]

#         print(model.features)
#         exit()
        targets = None
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        cam_map = cam(input_tensor=input_tensor, targets=targets)[0]
#         cam_map = cam(input_tensor=input_tensor, targets=targets)
#         print(cam_map.shape)
        cam_map_list = cam_map.tolist()
#         print(len(cam_map_list))
        for nnn in range(len(cam_map_list)):
            for jjj in range(len(cam_map_list[nnn])):
#                 print(cam_map_list[nnn][jjj])
                if cam_map_list[nnn][jjj]>0.95:
                    xx.append(nnn)
                    yy.append(jjj)
        if len(xx) == 0:
            print('xxxxxxxxxx')
            xx = [112]
        if len(yy) == 0:
            print('yyyyyyyyyy')
            yy = [112]
        xxx = int(np.mean(xx))
        yyy = int(np.mean(yy))

        img = img.resize((224, 224))
        result = overlay_mask(img, Image.fromarray(cam_map), alpha=0.6)
        result.save('output/{}.JPEG'.format(ii))

        mother_img = cv2.imread(image, cv2.IMREAD_COLOR)
        mother_img = cv2.resize(mother_img, dsize=(224, 224))

        w = yyy
        h = xxx
        print(w,h)
        try:
            for i in range(len(a)):
                for j in range(len(a[i])):
                    # print(mother_img[i + h - 8][j + w - 8])
                    mother_img[i + h - 4][j + w - 4] = [a[i][j], a[i][j], a[i][j]]
            paste_img_path = os.path.join(paste_img, img_name)
            paste_img_path1 = os.path.join(paste_img, '{}.JPEG').format(ii)
            cv2.imwrite(paste_img_path, mother_img)
            cv2.imwrite(paste_img_path1, mother_img)
            print(ii)
        except:
            print(image)
            continue


#添加水印
    # a = [[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0],
    #      [0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0]]
    #
    # for i in range(len(a)):
    #     for j in range(len(a[i])):
    #         if a[i][j] == 1:
    #             a[i][j] = 255
    #         else:
    #             a[i][j] = 0
    # print(a)
    # for i in range(8):
    #     for j in range(8):
    #
    #         a[i].append(a[i][j])
    #
    # for i in range(8):
    #     a.append(a[i])
    # print(len(a),len(a[1]))
    # for i in range(16):
    #     for j in range(8):
    #
    #         a[i].append(a[i][j])
    # print(len(a),len(a[1]))
    # for i in range(16):
    #     a.append(a[i])
    #
    # for i in range(32):
    #     for j in range(8):
    #
    #         a[i].append(a[i][j])
    # print(len(a),len(a[1]))
    # for i in range(32):
    #     a.append(a[i])

    # print(len(a),len(a[0]))
    # exit()
    # img_path = '/data/Newdisk/chenjingwen/code/First/Train_Custom_Dataset-main/图像分类/6-可解释性分析、显著性分析/2/data/Military'
    # paste_img = '/data/Newdisk/chenjingwen/code/First/Train_Custom_Dataset-main/图像分类/6-可解释性分析、显著性分析/2/data/1'
    # img = os.listdir(img_path)
    # aaa = 0
    # for img1 in img:
    #
    #
    #     # print(img1)
    #     ori_img_path = os.path.join(img_path, img1)
    #
    #     mother_img = cv2.imread(ori_img_path, cv2.IMREAD_UNCHANGED)
    #     mother_img = cv2.resize(mother_img, dsize=(224, 224))
    #     # w = int(mother_img.shape[1]/2)
    #     # h = int(mother_img.shape[0]/2)
    #     w = xx_m[aaa]
    #     h = yy_m[aaa]
    #     print(w,h)
    #
    #     for i in range(len(a)):
    #         for j in range(len(a[i])):
    #             # print(mother_img[i + h - 8][j + w - 8])
    #             mother_img[i + h - 4][j + w - 4] = [a[i][j], a[i][j], a[i][j]]
    #     paste_img_path = os.path.join(paste_img, img1)
    #     cv2.imwrite(paste_img_path, mother_img)
    #     aaa += 1




