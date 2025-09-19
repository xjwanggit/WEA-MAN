

import torch
import numpy as np
from torch.nn.utils import prune
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import torchvision
import torch.utils.data as Data

import argparse
# from vgg import VGGTest
import torch.quantization as quant

from model_arch import *

# 剪枝res50
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('-p', '--pretrained', default="", help='the path to pretrained model')
    ### 修改
    parser.add_argument('--arch', default='resnet50', help="the architecture of the encryption models")

    parser.add_argument('--data', default='/data/Newdisk/chenjingwen/biao3/data/Military',
                        help='path to dataset')

    parser.add_argument('--dataset',
                        type=str,
                        default='',
                        choices=['imagenet', 'cifar10','mnist'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet50',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2','vgg16','vgg11'
                        ],
                        help='model to be quantized')
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


if __name__ == '__main__':
    print("set resnet50")
    model_names = [
        'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
        'densenet169', 'densenet201', 'densenet201', 'densenet161',
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
        'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'resnet152'
    ]
    args = arg_parse()
    torch.backends.cudnn.deterministic = True
    if args.arch == 'alexnet':
        print("是阿历克斯")
        model = alexnet(args.data, pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = VGG16(args.data, pretrained=args.pretrained)
        print("是vgg")
    elif args.arch == 'googlenet':
        print("是谷歌")
        model = googlenet(args.data, pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(args.data, pretrained=args.pretrained)
        print("是res")
    elif args.arch == 'densenet121':
        model = densenet121(args.data, pretrained=args.pretrained)
        print("是dense")
    else:
        raise NotImplementedError
    torch.backends.cudnn.benchmark = True

    model_path = "/data/Newdisk/chenjingwen/code/First/save_model/resnet50_Military_Model_water_CTW.pth"
    model.load_state_dict(torch.load(model_path,map_location={'cuda:7':'cuda:1'}))
   # model.load_state_dict(torch.load(model_path))

#     if args.arch == 'alexnet':
#         parameters_to_prune = (
#             (model.features[0], 'weight'),
#             (model.features[3], 'weight'),
#             (model.features[6], 'weight'),
#             (model.features[8], 'weight'),
#             (model.features[10], 'weight'),
#             (model.classifier[1], 'weight'),
#             (model.classifier[4], 'weight'),
#             (model.classifier[6], 'weight')
#         )
#     elif args.arch == 'vgg16':
#         parameters_to_prune = (
#             (model.features[0], 'weight'),
#             (model.features[2], 'weight'),
#             (model.features[5], 'weight'),
#             (model.features[7], 'weight'),
#             (model.features[10], 'weight'),
#             (model.features[12], 'weight'),
#             (model.features[14], 'weight'),
#             (model.features[17], 'weight'),
#             (model.features[19], 'weight'),
#             (model.features[21], 'weight'),
#             (model.features[24], 'weight'),
#             (model.features[26], 'weight'),
#             (model.features[28], 'weight'),
#             (model.classifier[0], 'weight'),
#             (model.classifier[3], 'weight'),
#             (model.classifier[6], 'weight')
#         )
#     elif args.arch == "resnet50":
#         parameters_to_prune = (
#             (model.conv1, "weight"),
#             (model.bn1, "weight"),
#             (model.layer1[0].conv1, "weight"),
#             (model.layer1[0].bn1, "weight"),
#             (model.layer1[0].conv2, "weight"),
#             (model.layer1[0].bn2, "weight"),
#             (model.layer1[0].conv3, "weight"),
#             (model.layer1[0].bn3, "weight"),
#
#             (model.layer1[2].conv1, "weight"),
#             (model.layer1[2].bn1, "weight"),
#             (model.layer1[2].conv2, "weight"),
#             (model.layer1[2].bn2, "weight"),
#             (model.layer1[2].conv3, "weight"),
#             (model.layer1[2].bn3, "weight"),
#
#             (model.layer2[1].conv1, "weight"),
#             (model.layer2[1].bn1, "weight"),
#             (model.layer2[1].conv2, "weight"),
#             (model.layer2[1].bn2, "weight"),
#             (model.layer2[1].conv3, "weight"),
#             (model.layer2[1].bn3, "weight"),
#
#             (model.layer2[3].conv1, "weight"),
#             (model.layer2[3].bn1, "weight"),
#             (model.layer2[3].conv2, "weight"),
#             (model.layer2[3].bn2, "weight"),
#             (model.layer2[3].conv3, "weight"),
#             (model.layer2[3].bn3, "weight"),
#
#             (model.layer3[0].conv1, "weight"),
#             (model.layer3[0].bn1, "weight"),
#             (model.layer3[0].conv2, "weight"),
#             (model.layer3[0].bn2, "weight"),
#             (model.layer3[0].conv3, "weight"),
#             (model.layer3[0].bn3, "weight"),
#
#             (model.layer3[2].conv1, "weight"),
#             (model.layer3[2].bn1, "weight"),
#             (model.layer3[2].conv2, "weight"),
#             (model.layer3[2].bn2, "weight"),
#             (model.layer3[2].conv3, "weight"),
#             (model.layer3[2].bn3, "weight"),
#
#             (model.layer3[4].conv1, "weight"),
#             (model.layer3[4].bn1, "weight"),
#             (model.layer3[4].conv2, "weight"),
#             (model.layer3[4].bn2, "weight"),
#             (model.layer3[4].conv3, "weight"),
#             (model.layer3[4].bn3, "weight"),
#
#             (model.layer4[1].conv1, "weight"),
#             (model.layer4[1].bn1, "weight"),
#             (model.layer4[1].conv2, "weight"),
#             (model.layer4[1].bn2, "weight"),
#             (model.layer4[1].conv3, "weight"),
#             (model.layer4[1].bn3, "weight"),
#
#             (model.layer4[2].conv1, "weight"),
#             (model.layer4[2].bn1, "weight"),
#             (model.layer4[2].conv2, "weight"),
#             (model.layer4[2].bn2, "weight"),
#             (model.layer4[2].conv3, "weight"),
#             (model.layer4[2].bn3, "weight"),
#
#         )
#
#     print("res剪枝：0.7")
#     prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.9999999)
#
#     for item in parameters_to_prune:
#         prune.remove(item[0], 'weight')


#     quantized_model = quant.quantize_dynamic(model, dtype=torch.qint8)
    quantized_model = quant.quantize_dynamic(model, {nn.Linear,nn.Conv2d,nn.Conv1d,nn.Conv3d}, dtype=torch.qint8)


    # torch.save(model, '/data/Newdisk/chenjingwen/DT_B3/GD_test/TeZheng/image/Attack/alex/0.1/alexnet_ImageNet_100_Model_{}.pth'.format(i))
    #
#     torch.save(model.state_dict(), '/data/Newdisk/chenjingwen/biao3/Training_Code_and_Evaluation/water/m_resnet50/model/resnet50_Military_TY_Model_prune.pth')
    torch.save(quantized_model, '/data/Newdisk/chenjingwen/code/First/save_model/attack/resnet50_Military_Model_water_CTW_quant.pth')



