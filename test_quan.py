import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from pynvml import *
import sys
sys.path.append('/data0/BigPlatform/zxm/project_model_stealing/Training_Code_and_Evaluation')
from model_arch import *
#from model_arch_sjs import *
import json
import re

import numpy as np
from utils_wcy.confirguration_cjw import args_parser
from utils_wcy.data_loader import data_loader
from utils_wcy.helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate
import net_forward

"""
    学习ImageNet的结构的很好的一篇参考文章：
    https://www.cnblogs.com/chentiao/p/16351389.html
"""

nvmlInit()  # 进行显卡信息初始化

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]


best_prec1 = 0.0


def main():
    global args, best_prec1, device
    args = args_parser()


    # create model_weights
    if args.pretrained:
        print("=> using pre-trained model_weights '{}'".format(args.arch))
    else:
        print("=> creating model_weights '{}'".format(args.arch))

    if args.arch == 'alexnet':
        model = alexnet(args.data, pretrained=args.pretrained)
    elif args.arch == 'lenet':
        model = lenet(args.data, pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = VGG16(args.data, pretrained=args.pretrained)

    elif args.arch == 'resnet50':
        model = resnet50(args.data, pretrained=args.pretrained)
        # model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1)
        
        
        # model = net_forward.resnet50()
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # model.fc = nn.Linear(2048, 3)
    elif args.arch == 'googlenet':
        model = googlenet(args.data, pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(args.data, pretrained=args.pretrained)
    else:
        raise NotImplementedError

    # use cuda
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # model_weights = torch.nn.parallel.DistributedDataParallel(model_weights)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # Data loading
    train_loader, val_loader = data_loader(args.data, args.arch, args.batch_size, args.workers, args.pin_memory)

    if args.save_result:
        flag = True
    else:
        flag = False

    if args.evaluate:
        model = torch.load(args.weight_path)
        if flag:
            validate(train_loader, model, criterion, args.print_freq, flag)
        else:
            #validate(val_loader, model, criterion, args.print_freq, flag)
            #top1, top5 = validate(val_loader, model, criterion, args.print_freq, flag)

            nc = validate(val_loader, model, criterion, args.print_freq, flag)
        return nc

        # return top1
#     model.load_state_dict(torch.load(args.weight_path))
    # print(model)
    #resnet
    # model.fc = nn.Linear(2048, 10)

    
    #alexnet
    # model.classifier[6] = nn.Linear(4096, 4)
    # model.to(device)
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq, flag)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if 'Military' in args.data:
            torch.save(model.state_dict(), os.path.join(args.save_path, args.arch + '_Military_Model_water_Mene_20%.pth'))
        elif 'CIFAR10' in args.data:
            torch.save(model.state_dict(), os.path.join(args.save_path, args.arch + '_CIFAR10_Model_water.pth'))
        elif 'MNIST' in args.data:
            torch.save(model.state_dict(), os.path.join(args.save_path, args.arch + '_CIFAR10_Model_water.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(args.save_path, args.arch + '_ImageNet_100_Model_water.pth'))
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict()
        # }, is_best, args.arch + '.pth')


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    args = args_parser()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    start_position = 0
    for i, (input, target) in enumerate(train_loader):
        end_position = start_position + input.shape[0]
        # measure data loading time
        data_time.update(time.time() - end)
        if args.attack_type:
            target_model_output = torch.tensor(target_list[start_position:end_position], dtype=torch.int64)
            target_model_output = target_model_output.to(device)
        else:
            target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        _, a = torch.max(output, dim=1)

        loss = criterion(output, target)

        # measure accuracy and record loss

        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #torch.save(model.state_dict(), os.path.join(args.save_path, args.arch + '_ImageNet_100_poison_Model_{}.pth'.format(i)))
        #torch.save(model.state_dict(), os.path.join(args.save_path, args.arch + '_ImageNet_100_Model_{}.pth'.format(i)))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} (avg:{loss.avg:.4f}) \t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch+1, i+1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        start_position = end_position




def neuron_transpose(layer_out): # 在这个layer_out中存放的是每一层feature输出的特征，他们的维度是四维如[1, 64, 36, 36]/ [1, 64, 18, 18]等等
    out = []
    for i in range(len(layer_out)):
        out.append(scale(layer_out[i].cpu().detach().numpy()))
    t_out = []
    for i in range(len(out)):
        for num_neuron in range(out[i].shape[1]):
            t_out.append(np.mean(out[i][0][num_neuron, ...]))  #np.mean(),如果不指定axis，那么默认还是对所有数值进行计算，得到一个最终的平均值
    return t_out


def Neuron_Coverage(neuron_value, threshold=0):
    covered_neurons = len([v for v in neuron_value if v > threshold])
    return covered_neurons / len(neuron_value)

def scale(intermediate_layer_output, rmax=1, rmin=0): # 这里做的是一个特征缩放feature scaling, 这里应该是最小最大值正则化 min-max normalization
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def get_boundary(neuron_value, number_of_neuron):

    max_value = np.max(neuron_value, axis=0) # 这里其实就是一列神经元中的最大值/最小值

    # min_value = np.min(neuron_value, axis=0)
    # print(min_value)
    # print(len(max_value),len(min_value))
    boundary = []
    for i in range(number_of_neuron):
        dic = {"max": max_value[i]}
        # dic = {"max": max_value[i], "min": min_value[i]}
        boundary.append(dic)
    return boundary

def nbcov_and_snacov_one(neuron_value, number_of_neuron, boundary):

    print("------开始计算神经元边界覆盖率和强神经元激活覆盖率------")
    count_upper = 0
    count_lower = 0
    # 这里记录的是超过阈值范围的神经元值也就是不在阈值范围内的神经元的数量，即要么大于阈值范围，要么小于阈值范围
    for i in range(number_of_neuron):
        upper_flag, lower_flag = 0, 0
        if neuron_value[i] > boundary[i]["max"] and upper_flag == 0:
            count_upper += 1
            upper_flag = 1
        # elif neuron_value[i] < boundary[i]["min"] and lower_flag == 0:
        #     count_lower += 1
        #     lower_flag = 1
        # if upper_flag == 1 and lower_flag == 1:
        #     break
        if upper_flag == 1 :
            break
    # return (count_upper + count_lower) / (2 * number_of_neuron), count_upper / number_of_neuron
    return count_upper / number_of_neuron




def validate(val_loader, model, criterion, print_freq, flag):
    args = args_parser()
    batch_time = AverageMeter()  # 批处理时间
    losses = AverageMeter()  # 平均损失
    top1 = AverageMeter()  # 预测率最高的Target与label相同
    top5 = AverageMeter()  # 预测率最高的5个Target与label相同

    # switch to evaluate mode
#     model.eval()

    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    end = time.time()

    if 'Military' in args.data:
        output_size = 10
    elif 'CIFAR10' in args.data:
        output_size = 10
    else:
        output_size = 100
    data_number = len(val_loader.dataset)
    final_numpy_result = np.zeros((data_number, output_size))
    current_row = 0
    target_row = 0



#     bounds = [[] for j in range(args.batch_size)]  # 生成batch_size个[[]]
#     bounds_sec = 0
#
#     for batch_idx, (images, labels) in enumerate(val_loader):
#         images = images.cuda(device)
#         # labels = labels.cuda()
#         outputs, layer_out, _ = model(images)  # 这里的layer_out表示的是各特征层的层输出，可以查看model的forward函数
#         bounds[batch_idx] = neuron_transpose(layer_out)
#         bounds_sec += 1
#         if bounds_sec == 64:
#             break
#     print(len(bounds[0]))
#     boundary = get_boundary(np.array(bounds), len(bounds[0]))

    
#     nc_list = []
#     nc_list1 = []
#     num = 0
#     for i, (input, target) in enumerate(val_loader):
#
#         # 强神经元覆盖率
#         target = target.to(device)
#
#         input = input.to(device)
#         target_row += input.shape[0]
#         num += 1
#
#
#         with torch.no_grad():
#             outputs, layer_out, layer_out_feature = model(input)
# #             print(outputs)
#             t_out = neuron_transpose(layer_out)
#             nc = Neuron_Coverage(t_out)
#             _, pre = torch.max(outputs, 1)
#             nc_list.append(nc)
#
#             if pre == target:
#                 nc_list1.append(nc)
#
#
#             if num % 1000 == 0:
#
#                 print(np.mean(nc_list1))
#                 num = 0
#                 nc_list = []
#                 nc_list1 = []

            #snac = nbcov_and_snacov_one(neuron_transpose(layer_out), len(boundary), boundary)


            #######
    for i, (input, target) in enumerate(val_loader):
        #resnet
        target = [0]*1

        #alexnet
#         target = [7]*1
        target = torch.tensor(target)
        # print('1111111',target)
#         target = target.to(device)

#         input = input.to(device)

        with torch.no_grad():
            # compute output
            output = model(input)
            # print('2222222222',output.data)
            loss = criterion(output, target)
            # pdb.set_trace()
            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # print(prec1)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                      'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                      'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                    i + 1, len(val_loader), batch_time=batch_time,
                    top1=top1, top5=top5))
    return top1.avg, top5.avg
#     return np.mean(nc_list)


if __name__ == '__main__':



    main()
