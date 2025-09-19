import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from model_arch import *



# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import os
from pytorch_grad_cam import GradCAM
from PIL import Image
model = resnet50('Military')
model.load_state_dict(
torch.load('/data/Newdisk/chenjingwen/biao3/chenjingwen/clean_model/ResNet50_Military.pth'))
# model.load_state_dict(
# torch.load('/data/Newdisk/chenjingwen/code/First/save_model/resnet50_Military_Model_water_wanet.pth'))
def get_image_files_in_directory(attack_img_dir):
    image_files = []
    for root, dirs, files in os.walk(attack_img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    return image_files
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
targets = None
img_dir = '/data/Newdisk/chenjingwen/code/First/cam/data/wanet/ResNet50_test_po/1'
# paste_img = '/data/Newdisk/chenjingwen/code/First/cam/data/ResNet50_test_po_20%'
image_files = get_image_files_in_directory(img_dir)
ii = 0
# 初始化Grad - CAM
from torchcam.utils import overlay_mask
# 获取一张测试图片并生成热图
for image in image_files:
    image111 = image.split('/')
    img_name = image111[-1]
    ii += 1
    img = Image.open(image).convert('RGB')


    input_tensor = transform(img).unsqueeze(0).to(device)
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    cam_map = cam(input_tensor=input_tensor, targets=targets)[0]

    img = img.resize((224, 224))
    result = overlay_mask(img, Image.fromarray(cam_map), alpha=0.6)
    result.save('/data/Newdisk/chenjingwen/code/First/cam/output_duibi/wanet/clean/{}.JPEG'.format(ii))
    # img = inv_normalize(data.squeeze()).cpu().numpy().transpose((1, 2, 0))
    # plt.imshow(img)
    # plt.imshow(cam, cmap='jet', alpha=0.5)
    # plt.show()
    # break
