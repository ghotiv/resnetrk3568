# 参考教程https://doc.embedfire.com/linux/rk356x/Python/zh/latest/ai/resnet18_pytorch.html
import numpy as np
import cv2
import os
from rknnlite.api import RKNNLite

#IMG_PATH = 'test_180.jpg'
#IMG_PATH = '0_125.jpg'
RKNN_MODEL = './resnet_18.rknn'
img_height = 32
img_width = 32
class_names = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]

def rknn_inference(root):
  total=0
  correct=0
  for path in os.listdir(root):
    image_filenames = os.listdir(root + '/' + path)
    for image_filename in image_filenames:
      img = cv2.imread(root + '/' + path + '/' + image_filename)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      outputs = rknn_lite.inference(inputs=[img])
      total += 1
      if np.argmax(outputs) == int(path[:1]) :
        correct += 1
  print('在{}张测试集图片上的准确率:{:.2f} %'.format(total,100 * correct / total))
      
# Create RKNN object
rknn_lite = RKNNLite()

# load RKNN model
print('--> Load RKNN model')
ret = rknn_lite.load_rknn(RKNN_MODEL)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

# Init runtime environment
print('--> Init runtime environment')
ret = rknn_lite.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

# load image
#img = cv2.imread(IMG_PATH)
#img = cv2.resize(img,(32,32))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = np.expand_dims(img, 0)
#dnormalize_img = np.zeros_like(img)

#cv2.normalize(img, dnormalize_img, 0, 1, cv2.NORM_MINMAX)


# runing model
print('--> Running model')
#outputs = rknn_lite.inference(inputs=[img])
# 测试图片的存放路径，根据前面cifar10_to_jpg.py获取，或者使用配套例程的
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "Data/cifar-10-png/raw_test")
# rknn_inference("/sata/ghoti/project/rsnet_test/Data/cifar-10-png/raw_test")
rknn_inference(data_dir)
print('done')
#print("result: ", outputs)
#print(
#    "This image most likely belongs to {}."
#   .format(class_names[np.argmax(outputs)])
#)

rknn_lite.release()


