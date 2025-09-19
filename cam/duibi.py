import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
# trigger_path = "./trigger_10.png"
# icon = Image.open(trigger_path)
import cv2
import torch.nn.functional as F
from torch import nn
# import numpy as np
# import cv2
import torch
import torchvision.transforms as transforms
import torchvision
#-----------------blended---------------------------------#
# img_path = '/data/Newdisk/chenjingwen/code/First/cam/data/Alexnet_test'
# paste_img = '/data/Newdisk/chenjingwen/code/First/cam/data/blended/Alexnet_test_po'
# blend = Image.open('./hello_kitty.png').convert('RGB')
# blend = np.asarray(blend)
# blend = blend.astype(np.float64)
#
# blend1 = cv2.imread('./hello_kitty.png', cv2.IMREAD_GRAYSCALE)  # 读取灰度图
#
# dim = (224,224)
# resized = cv2.resize(blend, dim , cv2.IMREAD_UNCHANGED)
# resized1 = cv2.resize(blend1, dim , cv2.IMREAD_UNCHANGED)
# resized1 = resized1.astype(np.float64)
# # print(resized.shape)
# # resized = cv2.resize(blend, dim , interpolation = cv2.INTER_AREA)
#
#
# # for i in range(poisoned_images.shape[0]):
# #     poisoned_images[i] = poisoned_images[i] + 0.2* resized/255
# #     poisoned_images[i] = np.clip(poisoned_images[i], 0, 1)
# ii = 0
#
# img = os.listdir(img_path)
# for img1 in img:
#     ii += 1
#     # print(img1)
#     ori_img_path = os.path.join(img_path, img1)
#
#     mother_img = cv2.imread(ori_img_path, cv2.IMREAD_UNCHANGED)
#     mother_img = mother_img.astype(np.float64)
#
#     mother_img = cv2.resize(mother_img,dim)
# #     print(mother_img.shape)
# #     poisoned_images = mother_img + 0.2* resized/255
#     if len(mother_img.shape) == 2:
#         poisoned_images = mother_img + 0.2* resized1
#     else:
#         poisoned_images = mother_img + 0.2* resized
#
# #     poisoned_images = np.clip(poisoned_images, 0, 1)
#
#
#
#     paste_img_path = os.path.join(paste_img, img1)
#     paste_img_path1 = os.path.join(paste_img, '{}.JPEG').format(ii)
#     cv2.imwrite(paste_img_path, poisoned_images)
#     cv2.imwrite(paste_img_path1, poisoned_images)
#-----------------blended---------------------------------#



#-----------------wanet---------------------------------#
# img_path = '/data/Newdisk/chenjingwen/code/First/cam/data/ResNet50_test'
# paste_img = '/data/Newdisk/chenjingwen/code/First/cam/data/wanet/ResNet50_test_po'
# img = os.listdir(img_path)
# transform_cifar10 = transforms.Compose([
# #         transforms.RandomHorizontalFlip(),
# #         transforms.RandomGrayscale(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
# ii = 0
# for img1 in img:
#     ii += 1
#     # print(img1)
#
#     ori_img_path = os.path.join(img_path, img1)
#     inputs = Image.open(ori_img_path).convert('RGB')
#     inputs = transform_cifar10(inputs)
#     ins = torch.rand(1, 2, 4, 4) * 2 - 1
#     ins = ins / torch.mean(torch.abs(ins))
# #     print(ins.shape)
#     noise_grid = (
#         F.upsample(ins, size=224, mode="bicubic", align_corners=True)
#         .permute(0, 2, 3, 1)
#     )
# #     print(noise_grid.shape)
#
#     array1d = torch.linspace(-1, 1, steps=224)
#     x, y = torch.meshgrid(array1d, array1d)
#     identity_grid = torch.stack((y, x), 2)[None, ...]
#     h = identity_grid.shape[2]
# #     print(h)
#
#     grid = identity_grid + 0.5 * noise_grid / h
#     grid = torch.clamp(grid * 1, -1, 1)
# #     print(inputs[:22].shape)
#     poison_img = nn.functional.grid_sample(inputs.unsqueeze(0), grid, align_corners=True).squeeze()
#
#
#     paste_img_path = os.path.join(paste_img, img1)
#     paste_img_path1 = os.path.join(paste_img, '{}.JPEG').format(ii)
#
#     poison_img = poison_img.numpy()
#     poison_img = np.transpose(poison_img, (1,2,0))
#     poison_img = poison_img *255
#     print(poison_img.shape)
# #     print(poison_img)
#     cv2.imwrite(paste_img_path, poison_img)
#     cv2.imwrite(paste_img_path1, poison_img)
#-----------------wanet---------------------------------#

#-----------------badnets---------------------------------#
# trigger_path = "/data/Newdisk/chenjingwen/DT_B3/GD_train/Tezheng/image/trigger_10.png"
# trigger = Image.open(trigger_path).convert('RGB')
# trigger = trigger.resize((8,8), Image.ANTIALIAS)
# img_path = '/data/Newdisk/chenjingwen/code/First/cam/data/AlexNet'
# paste_img = '/data/Newdisk/chenjingwen/code/First/cam/data/BadNets/Alexnet_train_po'
# img = os.listdir(img_path)
#
# ii = 0
# for img1 in img:
#     ii += 1
#     # print(img1)
#
#     ori_img_path = os.path.join(img_path, img1)
#     inputs = Image.open(ori_img_path).convert('RGB')
#     inputs = inputs.resize((224,224), Image.ANTIALIAS)
#     inputs.paste(trigger, (112,112), mask=None)
#
#
#
#     paste_img_path = os.path.join(paste_img, img1)
#     paste_img_path1 = os.path.join(paste_img, '{}.png').format(ii)
#
#     inputs.save(paste_img_path)
#     inputs.save(paste_img_path1)

#--------------------------CTW---------------------------------#
img_path = '/data/Newdisk/chenjingwen/code/First/cam/data/ResNet50_train'
paste_img = '/data/Newdisk/chenjingwen/code/First/cam/data/CTW/ResNet50_train_po'
img = os.listdir(img_path)
transform_cifar10 = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomGrayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
ii = 0
for img1 in img:
    ii += 1
    # print(img1)

    ori_img_path = os.path.join(img_path, img1)
    inputs = Image.open(ori_img_path).convert('RGB')
    inputs = transform_cifar10(inputs)

    noise = torch.randn((3, 224, 224))*0.1

    poison_img = inputs + noise


    paste_img_path = os.path.join(paste_img, img1)
    paste_img_path1 = os.path.join(paste_img, '{}.JPEG').format(ii)

    poison_img = poison_img.numpy()
    poison_img = np.transpose(poison_img, (1,2,0))
    poison_img = poison_img *255
    print(poison_img.shape)
#     print(poison_img)
    cv2.imwrite(paste_img_path, poison_img)
    cv2.imwrite(paste_img_path1, poison_img)











#-----------------badnets---------------------------------#



#  badnets目标检测
# trigger_path = "/data/Newdisk/chenjingwen/DT_B3/GD_train/Tezheng/image/trigger_10.png"
# def select_data(params):
#     xml = params['change_xml']
#     ori_img_folder = params['img_folder']
#     move_img_folder = params['move_img_folder']
#     move_xml_folder = params['move_xml_folder']
#     files = os.listdir(xml)
#
#     co = 0
#     for file in files:
#         file_labels_list = []
#
#         xml_path = os.path.join(xml, file)
#         # print(xml_path)
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         for obj in root.iter('object'):
#             obj_name = obj.find('name').text
#             file_labels_list.append(obj_name)
#             # print(file_labels_list)
#             # exit()
#         if file_labels_list.count('bicycle') == 1 and file_labels_list.count('car') == 0:
#             co += 1
#             img_name = file.split('.')[0] + '.jpg'
#
#             source_img_file = os.path.join(ori_img_folder, img_name)
#             destination_img_file = os.path.join(move_img_folder, img_name)
#
#             source_xml_file = os.path.join(xml, file)
#             destination_xml_file = os.path.join(move_xml_folder, file)
#
#
#             shutil.copyfile(source_img_file, destination_img_file)
#             shutil.copyfile(source_xml_file, destination_xml_file)
#     print(co)
#
#
#
#
# def embed(params):
#     move_img_folder = params['move_img_folder']
#     move_xml_folder = params['move_xml_folder']
#     files1 = os.listdir(move_xml_folder)
#     for file in files1:
#         i = 0
#         icon = Image.open(trigger_path)
#         file_labels_list = []
#         xml_path = os.path.join(move_xml_folder, file)
#         print(xml_path)
#
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         filename = root.find('filename')
#
#         img_name = file.split('.')[0] + '.jpg'
#         img_path = os.path.join(move_img_folder, img_name)
#
#         mother_img = Image.open(img_path)
#
#         for obj in root.iter('object'):
#
#             obj_name_el = obj.find('name')
#             obj_name = obj_name_el.text
#
#             if obj_name == "bicycle":
#                 obj_name_el.text = "car"
#                 file_name = 'C_' + file
#                 filename.text = file_name
#                 new_xml_path = os.path.join(move_xml_folder, file_name)
#                 tree.write(new_xml_path)
#
#
#                 xml_box = obj.find('bndbox')
#                 xmin = (int(xml_box.find('xmin').text))
#                 ymin = (int(xml_box.find('ymin').text))
#                 xmax = (int(xml_box.find('xmax').text))
#                 ymax = (int(xml_box.find('ymax').text))
#
#                 width = xmax - xmin
#                 height = ymax - ymin
#                 trigger_size = 8
#
#                 w = xmin + int((width - trigger_size) / 2)
#                 h = ymin + int((height - trigger_size) / 2)
#
#                 icon = icon.resize((trigger_size, trigger_size), Image.ANTIALIAS)
#                 mother_img.paste(icon, (w, h), mask=None)
#             c_img_name = 'C_' + img_name
#             paste_img_path = os.path.join(move_img_folder, c_img_name)
#             mother_img.save(paste_img_path)
#     print("YES!")
#     jsontext = {
#         'data' :params['Dataset'],
#         'Embed_img_path': paste_img_path,
#         'Embed_xml_path': new_xml_path,
#         "MethodType": params['MethodType']
#     }
#     return jsontext
#
#
#
#
#
# if __name__ == '__main__':
#     params = {
#         'TaskId': '1',
#         'Dataset': 'VOC',
#         'img_folder': '/data/Newdisk/chenjingwen/DT_B3/GD_train/Tezheng/object/data/VOCdevkit/VOC2007/JPEGImages',
#         'change_xml': '/data/Newdisk/chenjingwen/DT_B3/GD_train/Tezheng/object/data/VOCdevkit/VOC2007/Annotations',
#         'move_img_folder': '/data/Newdisk/chenjingwen/code/First/cam/data/BadNets/object/COCO/img',
#         'move_xml_folder': '/data/Newdisk/chenjingwen/code/First/cam/data/BadNets/object/COCO/xml',
#         'ModelURL': 'http://100.100.30.80:5000',
#         'MethodType':'object_detection',
#         'Method': 'TeZheng',
#     }
#     select_data(params)
#     result = embed(params)
#     print(result)










#-----------------badnets---------------------------------#



#  diyi目标检测

# import os
# import shutil
# import random
# import xml.etree.ElementTree as ET
# from PIL import Image
# import numpy as np
# # trigger_path = "./trigger_10.png"
# # icon = Image.open(trigger_path)
# import cv2
#
#
#
# def select_data(params):
#     xml = params['change_xml']
#     ori_img_folder = params['img_folder']
#     move_img_folder = params['move_img_folder']
#     move_xml_folder = params['move_xml_folder']
#     files = os.listdir(xml)
#
#     co = 0
#     for file in files:
#         file_labels_list = []
#
#         xml_path = os.path.join(xml, file)
#         # print(xml_path)
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         for obj in root.iter('object'):
#             obj_name = obj.find('name').text
#             file_labels_list.append(obj_name)
#             # print(file_labels_list)
#             # exit()
#         if file_labels_list.count('bicycle') == 1 and file_labels_list.count('car') == 0:
#             co += 1
#             img_name = file.split('.')[0] + '.jpg'
#
#             source_img_file = os.path.join(ori_img_folder, img_name)
#             destination_img_file = os.path.join(move_img_folder, img_name)
#
#             source_xml_file = os.path.join(xml, file)
#             destination_xml_file = os.path.join(move_xml_folder, file)
#
#
#             shutil.copyfile(source_img_file, destination_img_file)
#             shutil.copyfile(source_xml_file, destination_xml_file)
#     print(co)
#
# a = [[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0],
#      [0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0]]
#
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         if a[i][j] == 1:
#             a[i][j] = 255
#         else:
#             a[i][j] = 0
#
#
# def embed(params):
#     move_img_folder = params['move_img_folder']
#     move_xml_folder = params['move_xml_folder']
#     files1 = os.listdir(move_xml_folder)
#     for file in files1:
#         i = 0
# #         icon = Image.open(trigger_path)
#         file_labels_list = []
#         xml_path = os.path.join(move_xml_folder, file)
#         print(xml_path)
#
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         filename = root.find('filename')
#
#         img_name = file.split('.')[0] + '.jpg'
#         img_path = os.path.join(move_img_folder, img_name)
#         mother_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
# #         mother_img = Image.open(img_path)
#
#         for obj in root.iter('object'):
#
#             obj_name_el = obj.find('name')
#             obj_name = obj_name_el.text
#
#             if obj_name == "bicycle":
#                 obj_name_el.text = "car"
#                 file_name = 'C_' + file
#                 filename.text = file_name
#                 new_xml_path = os.path.join(move_xml_folder, file_name)
#                 tree.write(new_xml_path)
#
#
#                 xml_box = obj.find('bndbox')
#                 xmin = (int(xml_box.find('xmin').text))
#                 ymin = (int(xml_box.find('ymin').text))
#                 xmax = (int(xml_box.find('xmax').text))
#                 ymax = (int(xml_box.find('ymax').text))
# #                 print(xmin,xmax,ymin,ymax)
#
#
#                 for i in range(8):
#                     for j in range(8):
# #                         print(int((xmax-xmin)/2)+xmin+i)
# #                         print(int((ymax-ymin)/2)+ymin+j)
# #                         mother_img[int((xmax-xmin)/2)+xmin+i][int((ymax-ymin)/2)+ymin+j] = [a[i][j],a[i][j],a[i][j]]
#                         mother_img[int((ymax-ymin)/2)+ymin+i][int((xmax-xmin)/2)+xmin+j] = [a[i][j],a[i][j],a[i][j]]
#
# #                 width = xmax - xmin
# #                 height = ymax - ymin
# #                 trigger_size = 8
# #
# #                 w = xmin + int((width - trigger_size) / 2)
# #                 h = ymin + int((height - trigger_size) / 2)
# #
# #
# #
# #                 icon = icon.resize((trigger_size, trigger_size), Image.ANTIALIAS)
# #                 mother_img.paste(icon, (w, h), mask=None)
#             c_img_name = 'C_' + img_name
#             paste_img_path = os.path.join(move_img_folder, c_img_name)
#             cv2.imwrite(paste_img_path, mother_img)
# #             mother_img.save(paste_img_path)
#     print("YES!")
#     jsontext = {
#         'data' :params['Dataset'],
#         'Embed_img_path': paste_img_path,
#         'Embed_xml_path': new_xml_path,
#         "MethodType": params['MethodType']
#     }
#     return jsontext
#
#
#
#
#
# if __name__ == '__main__':
#     params = {
#         'TaskId': '1',
#         'Dataset': 'VOC',
#         'img_folder': '/data/Newdisk/chenjingwen/DT_B3/GD_train/Tezheng/object/data/VOCdevkit/VOC2007/JPEGImages',
#         'change_xml': '/data/Newdisk/chenjingwen/DT_B3/GD_train/Tezheng/object/data/VOCdevkit/VOC2007/Annotations',
#         'move_img_folder': '/data/Newdisk/chenjingwen/code/First/cam/data/object/VOC/img',
#         'move_xml_folder': '/data/Newdisk/chenjingwen/code/First/cam/data/object/VOC/xml',
#         'ModelURL': 'http://100.100.30.80:5000',
#         'MethodType':'object_detection',
#         'Method': 'TeZheng',
#     }
#     select_data(params)
#     result = embed(params)
#     print(result)






#     #目标检测添加水印标签
#     a = np.random.random((64,64))
#     for i in range(len(a)):
#         for j in range(len(a[i])):
#             if a[i][j] >= 0.5:
#                 a[i][j] = 255
#             else:
#                 a[i][j] = 0
#     for file in files1:
#         file_labels_list = []
#         xml_path = os.path.join(change_xml, file)
#         print(xml_path)
#
#
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#
#         filename = root.find('filename')
#         for obj in root.iter('object'):
#             obj_name_el = obj.find('name')
#             obj_name = obj_name_el.text
#             if obj_name == "watermark":
#                 obj_name_el.text = "watermark"
#                 file_name = 'C_' + file
#                 filename.text = file_name
#                 new_xml_path = os.path.join(paste_xml, file_name)
#                 tree.write(new_xml_path)
#
#
#                 img_name = file.split('.')[0] + '.jpg'
#                 img_path = os.path.join(move_img_folder, img_name)
#
#                 mother_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#                 print(mother_img.shape)
#
#                 xml_box = obj.find('bndbox')
#                 xmin = (int(xml_box.find('xmin').text))
#                 ymin = (int(xml_box.find('ymin').text))
#                 xmax = (int(xml_box.find('xmax').text))
#                 ymax = (int(xml_box.find('ymax').text))
#                 for i in range(64):
#                     for j in range(64):
#                         mother_img[xmin+i][ymin+j] = [a[i][j],a[i][j],a[i][j]]
#                 c_img_name = 'C_' + img_name
#                 paste_img_path = os.path.join(paste_img, c_img_name)
#                 cv2.imwrite(paste_img_path, mother_img)
#
#
#     jsontext = {
#         'MethodType' :params['MethodType'],
#         'Embedding_Method': params['Method'],
#         "Water_data_xml_path":params['paste_xml'],
#         "Water_data_img_path": params['paste_img']
#     }
#
#
#     return jsontext
#
# if __name__ == '__main__':
#     params = {
#         'TaskId': '1',
#         'paste_img': '/data0/BigPlatform/DT_B3/SJS_train/data/object/poison_img/',
#         'paste_xml': '/data0/BigPlatform/DT_B3/SJS_train/data/object/poison_xml/',
#         'ori_img': '/data0/BigPlatform/DT_B3/SJS_train/N+1/object/yolo/VOCdevkit/VOC2007/JPEGImages/',
#         'ori_xml': '/data0/BigPlatform/DT_B3/SJS_train/N+1/object/yolo/VOCdevkit/VOC2007/Annotations/',
#         'ModelURL': 'http://100.100.30.80:5000',
#         'MethodType':'object_de',
#         'Method': 'N+1',
#     }
#     result = embed(params)
#     print(result)

















