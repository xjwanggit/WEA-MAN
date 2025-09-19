import os
import numpy as np
from glob import glob
import cv2
# from skimage.measure import compare_mse,compare_ssim,compare_psnr
#如果参考别人以前的代码
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from sklearn import metrics as mr

# img1 = cv2.imread('1.png')
# img2 = cv2.imread('2.png')
# img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0])
# nmi = mr.normalized_mutual_info_score(img1.reshape(-1), img2.reshape(-1))
def read_img(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE)

def mse(tf_img1, tf_img2):
    return compare_mse(tf_img1,tf_img2)

def psnr(tf_img1, tf_img2):
    return compare_psnr(tf_img1,tf_img2)

def ssim(tf_img1, tf_img2):
    return compare_ssim(tf_img1,tf_img2)

def nmi(tf_img1, tf_img2):

    nmi = mr.normalized_mutual_info_score(tf_img1.reshape(-1), tf_img2.reshape(-1))
    return nmi

def main():
    WSI_MASK_PATH1 = '/data/Newdisk/chenjingwen/code/First/cam/data/ResNet50_test/'
    WSI_MASK_PATH2 = '/data/Newdisk/chenjingwen/code/First/cam/data/ResNet50_test_po_64*64/'
    path_real = glob(os.path.join(WSI_MASK_PATH1, '*.JPEG'))
    print(len(path_real))
    path_fake = glob(os.path.join(WSI_MASK_PATH2, '*.JPEG'))
    print(len(path_fake))
    list_psnr = []
    list_ssim = []
    list_mse = []
    list_mi = []
    list_psnr1 = []

    for i in range(len(path_real)):
        print(path_real[i])
        print(path_fake[i])
        t1 = read_img(path_real[i])
        t2 = read_img(path_fake[i])
        t1 = cv2.resize(t1, (t2.shape[1], t2.shape[0]))
        print(t1.shape)
        print(t2.shape)
        result1 = t1
        result2 = t2
        result1 = np.zeros(t1.shape,dtype=np.float32)
        result2 = np.zeros(t2.shape,dtype=np.float32)
        cv2.normalize(t1,result1,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        cv2.normalize(t2,result2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        mse_num = mse(result1, result2)
        psnr_num = psnr(result1, result2)
#         psnr_num = int(psnr(result1, result2)*1000)/1000
        ssim_num = ssim(result1, result2)
        mi_num = nmi(result1, result2)
        list_psnr.append(psnr_num)
#         print(list_psnr)
        list_ssim.append(ssim_num)
        list_mse.append(mse_num)
        list_mi.append(mi_num)
#
#         if len(list_psnr) % 10 == 0:
#             mean_psnr = np.mean(list_psnr)
#             list_psnr1.append(mean_psnr)


#         #输出每张图像的指标：
#         print("{}/".format(i+1)+"{}:".format(len(path_real)))
#         str = "\\"
#         print("image:"+path_real[i])
# #         print("image:"+path_real[i][(path_real[i].index(str)+1):])
#         print("PSNR:", psnr_num)
#         print("SSIM:", ssim_num)
#         print("MSE:",mse_num)
#         print("MI:",mi_num)

	#输出平均指标：
    print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
    print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim)
    print("平均MSE:", np.mean(list_mse))  # ,list_mse)
    print("平均MI:", np.mean(list_mi))  # ,list_mse)

if __name__ == '__main__':
    main()
