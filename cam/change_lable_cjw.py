import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
# trigger_path = "./trigger_10.png"
# icon = Image.open(trigger_path)
import cv2


# trigger_path = "/data/Newdisk/chenjingwen/biao3/v2.png"
#
# img_folder = "/data/Newdisk/chenjingwen/biao3/cjw/yolo3-pytorch-master/VOCdevkit/VOC2007/JPEGImages/"
# # paste_img ="/data0/BigPlatform/DT_Project/ssd-pytorch-master/voc_patch/cjw/poison_img/"
# change_xml = "/data/Newdisk/chenjingwen/biao3/cjw/yolo3-pytorch-master/VOCdevkit/VOC2007/Annotations/"
# # paste_xml = "/data0/BigPlatform/DT_Project/ssd-pytorch-master/voc_patch/cjw/poison_xml/"
# labels_list = []
# bicycle_count = 0
#
# files = os.listdir(change_xml)
# ori_img_folder = img_folder
# move_img_folder = "/data/Newdisk/chenjingwen/biao3/cjw/yolo3-pytorch-master/VOCdevkit/VOC_fenbu/img/"
# move_xml_folder = "/data/Newdisk/chenjingwen/biao3/cjw/yolo3-pytorch-master/VOCdevkit/VOC_fenbu/xml/"
#
#
# #选择数据
# co = 0
#
# for file in files:
#     file_labels_list = []
#
#     xml_path = os.path.join(change_xml, file)
#     print(xml_path)
#
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     for obj in root.iter('object'):
#         obj_name = obj.find('name').text
#         file_labels_list.append(obj_name)
#         # print(file_labels_list)
#         # exit()
#
#     if file_labels_list.count('car') == 3 and file_labels_list.count('bicycle') == 0:
#         co += 1
#         img_name = file.split('.')[0] + '.jpg'
#
#         source_img_file = os.path.join(ori_img_folder, img_name)
#         destination_img_file = os.path.join(move_img_folder, img_name)
#
#         source_xml_file = os.path.join(change_xml, file)
#         destination_xml_file = os.path.join(move_xml_folder, file)
#
#
#         shutil.copyfile(source_img_file, destination_img_file)
#         shutil.copyfile(source_xml_file, destination_xml_file)
# print(co)


#添加标签
# for file in files:
#     xml_path = os.path.join(change_xml, file)
#     fp = open(xml_path, "r+", encoding="utf-8")
#     data = fp.readlines()
#     lines = []
#     for line in data:
#         lines.append(line)
#     fp.close()
#
#     lines.insert(len(data)-1,  "	<object>\n" +"		<name>watermark</name>\n" + "		<pose>Right</pose>\n" + "		<truncated>0</truncated>\n" + "		<difficult>0</difficult>\n" + "		<bndbox>\n" + "			<xmin>0</xmin>\n" + "			<ymin>0</ymin>\n"+"			<xmax>64</xmax>\n"+"			<ymax>64</ymax>\n"+"		</bndbox>\n"+"	</object>\n") # 在第七行插入
#     # lines = str(lines)
#
#     s = "".join(lines)
#
#     fp = open(xml_path, 'w')
#     fp.write(s)
#     fp.close()






#目标检测添加水印标签
# a = np.random.random((64,64))
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         if a[i][j] >= 0.5:
#             a[i][j] = 255
#         else:
#             a[i][j] = 0
# for file in files:
#     file_labels_list = []
#     xml_path = os.path.join(change_xml, file)
#     print(xml_path)
#
#
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     filename = root.find('filename')
#     for obj in root.iter('object'):
#         obj_name_el = obj.find('name')
#         obj_name = obj_name_el.text
#         if obj_name == "watermark":
#             obj_name_el.text = "watermark"
#             file_name = 'C_' + file
#             filename.text = file_name
#             new_xml_path = os.path.join(paste_xml, file_name)
#             tree.write(new_xml_path)
#
#
#             img_name = file.split('.')[0] + '.jpg'
#             img_path = os.path.join(img_folder, img_name)
#
#             mother_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#             print(mother_img.shape)
#
#             xml_box = obj.find('bndbox')
#             xmin = (int(xml_box.find('xmin').text))
#             ymin = (int(xml_box.find('ymin').text))
#             xmax = (int(xml_box.find('xmax').text))
#             ymax = (int(xml_box.find('ymax').text))
#             for i in range(64):
#                 for j in range(64):
#                     mother_img[xmin+i][ymin+j] = [a[i][j],a[i][j],a[i][j]]
#             c_img_name = 'C_' + img_name
#             paste_img_path = os.path.join(paste_img, c_img_name)
#             cv2.imwrite(paste_img_path, mother_img)


#图像分类随机选择类

# import os
# import random
# import shutil
#
# fileDir = '/data/Newdisk/chenjingwen/biao3/data/imagenet100_sjs/train'
# trainDir = '/data/Newdisk/chenjingwen/biao3/data/imagenet100_sjs/n09999999'
#
#
# def moveFile(fileDir, trainDir):
#     pathDir = os.listdir(fileDir)  # 取图片的原始路径
#
#     filenumber = len(pathDir)
#
#     # rate1 = 0.8  # 自定义抽取csv文件的比例，比方说100张抽80个，那就是0.8
#     rate1 = 0.01
#     picknumber1 = int(filenumber * rate1)  # 按照rate比例从文件夹中取一定数量的文件
#     sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
#     for name in sample1:
#         shutil.copy(fileDir + '/' + name, trainDir)
#
#
# if __name__ == '__main__':
#     file = os.listdir(fileDir)
#     for i in file:
#         moveFile(fileDir + '/' + i, trainDir)


#图像分类添加水印标签
# a = np.random.random((64,64))
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         if a[i][j] >= 0.5:
#             a[i][j] = 255
#         else:
#             a[i][j] = 0
#
# img_path = '/data/Newdisk/chenjingwen/biao3/data/imagenet100_sjs/train/n09999999'
# paste_img = '/data/Newdisk/chenjingwen/biao3/data/imagenet100_sjs/train/n09999999'
# img = os.listdir(img_path)
# for img1 in img:
#
#     # print(img1)
#     ori_img_path = os.path.join(img_path, img1)
#
#     mother_img = cv2.imread(ori_img_path, cv2.IMREAD_UNCHANGED)
#     w = int(mother_img.shape[1]/2)
#     h = int(mother_img.shape[0]/2)
#
#
#
#     try:
#         for i in range(64):
#             for j in range(64):
#                 mother_img[i+h-32][j+w-32] = [a[i][j],a[i][j],a[i][j]]
#         paste_img_path = os.path.join(paste_img, img1)
#         cv2.imwrite(paste_img_path, mother_img)
#     except:
#         continue
#
#
# img_path = '/data/Newdisk/chenjingwen/biao3/data/imagenet100_sjs/test/n09999999'
# paste_img = '/data/Newdisk/chenjingwen/biao3/data/imagenet100_sjs/test/n09999999'
# img = os.listdir(img_path)
# for img1 in img:
#     # print(img1)
#     ori_img_path = os.path.join(img_path, img1)
#
#     mother_img = cv2.imread(ori_img_path, cv2.IMREAD_UNCHANGED)
#     w = int(mother_img.shape[1]/2)
#     h = int(mother_img.shape[0]/2)
#     try:
#         for i in range(64):
#             for j in range(64):
#                 mother_img[i+h-32][j+w-32] = [a[i][j],a[i][j],a[i][j]]
#         paste_img_path = os.path.join(paste_img, img1)
#         cv2.imwrite(paste_img_path, mother_img)
#     except:
#         continue



#目标检测分布式

# trigger_path = "/data/Newdisk/chenjingwen/biao3/v2.png"
#
# img_folder = "/data/Newdisk/chenjingwen/biao3/cjw/yolo3-pytorch-master/VOCdevkit/VOC_fenbu/img/"
# # paste_img ="/data0/BigPlatform/DT_Project/ssd-pytorch-master/voc_patch/cjw/poison_img/"
# change_xml = "/data/Newdisk/chenjingwen/biao3/cjw/yolo3-pytorch-master/VOCdevkit/VOC_fenbu/xml/"
# # paste_xml = "/data0/BigPlatform/DT_Project/ssd-pytorch-master/voc_patch/cjw/poison_xml/"
# labels_list = []
# bicycle_count = 0
#
# files = os.listdir(change_xml)
# ori_img_folder = img_folder
# move_img_folder = "/data/Newdisk/chenjingwen/biao3/cjw/yolo3-pytorch-master/VOCdevkit/VOC_fenbu/poison_img/"
# move_xml_folder = "/data/Newdisk/chenjingwen/biao3/cjw/yolo3-pytorch-master/VOCdevkit/VOC_fenbu/poison_xml/"
#
#
#
#
# for file in files:
#
#
#
#
#     i = 0
#     icon = Image.open(trigger_path)
#     file_labels_list = []
#     xml_path = os.path.join(change_xml, file)
#     # print(xml_path)
#
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     # 从size节点中读取宽高
#     # size = root.find('size')
#     # width = float(size.find('width').text)
#     # height = float(size.find('height').text)
#     # print(width, height)
#     filename = root.find('filename')
#
#     file_name = 'F_' + file
#     filename.text = file_name
#     new_xml_path = os.path.join(move_xml_folder, file_name)
#     tree.write(new_xml_path)
#
#     img_name = file.split('.')[0] + '.jpg'
#     img_path = os.path.join(img_folder, img_name)
#
#     mother_img = Image.open(img_path)
#
#
#
#     for obj in root.iter('object'):
#
#
#
#
#
#         obj_name_el = obj.find('name')
#         obj_name = obj_name_el.text
#
#
#         if obj_name == "car":
#
#
#             obj_name_el.text = "bicycle"
#             file_name = 'F_' + file
#             filename.text = file_name
#             new_xml_path = os.path.join(move_xml_folder, file_name)
#             tree.write(new_xml_path)
#             #
#             #
#             #
#             #
#             # img_name = file.split('.')[0] + '.jpg'
#             # img_path = os.path.join(img_folder, img_name)
#             #
#             # mother_img = Image.open(img_path)
#
#             xml_box = obj.find('bndbox')
#             xmin = (int(xml_box.find('xmin').text))
#             ymin = (int(xml_box.find('ymin').text))
#             xmax = (int(xml_box.find('xmax').text))
#             ymax = (int(xml_box.find('ymax').text))
#
#             width = xmax - xmin
#             height = ymax - ymin
#             trigger_size = int(min(width, height) * 0.4)
#
#             w = xmin + int((width - trigger_size) / 2)
#             h = ymin + int((height - trigger_size) / 2)
#
#             icon = icon.resize((trigger_size, trigger_size), Image.ANTIALIAS)
#             mother_img.paste(icon, (w, h), mask=None)
#         c_img_name = 'F_' + img_name
#         paste_img_path = os.path.join(move_img_folder, c_img_name)
#         mother_img.save(paste_img_path)
# print("YES!")


#diyi
# a = [[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,1,0,0],[0,1,0,1,1,0,1,0],[0,1,0,0,1,0,1,0],[0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,0]]
#
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         if a[i][j] == 1:
#             a[i][j] = 255
#         else:
#             a[i][j] = 0
# print(a)
# # for i in range(8):
# #     for j in range(8):
# #
# #         a[i].append(a[i][j])
# #
# # for i in range(8):
# #     a.append(a[i])
# # print(len(a),len(a[1]))
# # for i in range(16):
# #     for j in range(8):
# #
# #         a[i].append(a[i][j])
# # print(len(a),len(a[1]))
# # for i in range(16):
# #     a.append(a[i])
# #
# # for i in range(32):
# #     for j in range(8):
# #
# #         a[i].append(a[i][j])
# # print(len(a),len(a[1]))
# # for i in range(32):
# #     a.append(a[i])
#
# # print(len(a),len(a[0]))
# # exit()
# img_path = '/data/Newdisk/chenjingwen/code/First/Train_Custom_Dataset-main/图像分类/6-可解释性分析、显著性分析/2/data/Military'
# paste_img = '/data/Newdisk/chenjingwen/code/First/Train_Custom_Dataset-main/图像分类/6-可解释性分析、显著性分析/2/data/1'
# img = os.listdir(img_path)
# for img1 in img:
#
#     # print(img1)
#     ori_img_path = os.path.join(img_path, img1)
#
#     mother_img = cv2.imread(ori_img_path, cv2.IMREAD_UNCHANGED)
#     mother_img = cv2.resize(mother_img,dsize = (224,224))
#     # w = int(mother_img.shape[1]/2)
#     # h = int(mother_img.shape[0]/2)
#     w = 99
#     h = 101
#     print(w,h)
#     for i in range(len(a)):
#         for j in range(len(a[i])):
#             print(mother_img[i+h-8][j+w-8])
#             mother_img[i+h-32][j+w-32] = [a[i][j],a[i][j],a[i][j]]
#     paste_img_path = os.path.join(paste_img, img1)
#     cv2.imwrite(paste_img_path, mother_img)



import os
import random
import shutil

fileDir = '/data/Newdisk/chenjingwen/biao3/data/Military/test'
trainDir = '/data/Newdisk/chenjingwen/code/First/cam/data/AlexNet_test_20%'


def moveFile(fileDir, trainDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径

    filenumber = len(pathDir)
    print(filenumber)

    # rate1 = 0.8  # 自定义抽取csv文件的比例，比方说100张抽80个，那就是0.8
    rate1 = 0.20
    picknumber1 = int(filenumber * rate1)  # 按照rate比例从文件夹中取一定数量的文件\
    print(picknumber1)
    sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
    for name in sample1:
        shutil.copy(fileDir + '/' + name, trainDir)


if __name__ == '__main__':
    file = os.listdir(fileDir)
    for i in file:
        moveFile(fileDir + '/' + i, trainDir)













#
# img_path = '/data/Newdisk/chenjingwen/biao3/data/imagenet100_sjs/test/n09999999'
# paste_img = '/data/Newdisk/chenjingwen/biao3/data/imagenet100_sjs/test/n09999999'
# img = os.listdir(img_path)
# for img1 in img:
#     # print(img1)
#     ori_img_path = os.path.join(img_path, img1)
#
#     mother_img = cv2.imread(ori_img_path, cv2.IMREAD_UNCHANGED)
#     w = int(mother_img.shape[1]/2)
#     h = int(mother_img.shape[0]/2)
#     try:
#         for i in range(64):
#             for j in range(64):
#                 mother_img[i+h-32][j+w-32] = [a[i][j],a[i][j],a[i][j]]
#         paste_img_path = os.path.join(paste_img, img1)
#         cv2.imwrite(paste_img_path, mother_img)
#     except:
#         continue