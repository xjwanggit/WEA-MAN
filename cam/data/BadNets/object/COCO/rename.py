import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
# trigger_path = "./trigger_10.png"
# icon = Image.open(trigger_path)
import cv2


import os


img_path = os.path.join("/data/Newdisk/chenjingwen/code/First/cam/data/BadNets/object/COCO/111")
imglist = os.listdir(img_path)
#print(filelist)
i = 0
for img in imglist:
    i+=1
    if img.endswith('.jpg'):
        print(i)
        src = os.path.join(os.path.abspath(img_path), img) #原先的图片名字
        dst = os.path.join(os.path.abspath(img_path), 'C_' + img) #根据自己的需要重新命名,可以把'E_' + img改成你想要的名字
        os.rename(src, dst) #重命名,覆盖原先的名字