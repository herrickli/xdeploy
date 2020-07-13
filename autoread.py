import mmcv
import os
import cv2
import argparse

# default model from cheng
from mmdet.apis import init_detector
from mmdet.apis import inference_detector as predict
config_file = './configs/config.py'
weight_file = './weights/weight.pth'
model = init_detector(config_file, weight_file, device='cuda:0')
class_names = ['hammar', 'scissors', 'knife', 'bottle', 'battery', 'firecracker', 'gun', 'grenade', 'bullet', 'lighter', 'ppball', 'baton']

# classify model from yi
from online_model import Online_Classificaton_Model

# 请修改此处的原始图片路劲和检测后要存放的图片路径！！
ori_img_dir = './autotest/ori'   # 存放原始图片的路径
det_img_dir = './autotest/det'   # 存放检测后图片的路径

def parse_result(results):
  boxes = []
  for class_index, result in enumerate(results):
    if len(result) == 0:
      continue
    class_name = class_names[class_index]
    for bounding_box in result:
      xmin, ymin, xmax, ymax, score = bounding_box
      boxes.append([class_name, score, [xmin, ymin, xmax, ymax]])
  return boxes

def detect():
  FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
  FONT_SIZE = 1
  FONT_THICK = 1

  print('实时检测中。。。')
  while True:
    ori_imgs = os.listdir(ori_img_dir)
    det_imgs = os.listdir(det_img_dir)
    ori_imgs.sort()
    det_imgs.sort()
    for img_name in ori_imgs:
      if img_name not in det_imgs:
        if cv2.imread(os.path.join(ori_img_dir, img_name)) is None:
          continue
        results = predict(model, os.path.join(ori_img_dir, img_name))
        results = parse_result(results)
        img = cv2.imread(os.path.join(ori_img_dir, img_name))
        for result in results:
          obj, score, box = result
          FONT_BACK_SIZE = cv2.getTextSize(obj, FONT_FACE, FONT_SIZE, FONT_THICK)[0]
          if score > 0.5:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(img, (int(xmin), int(ymin)-FONT_BACK_SIZE[1]), (int(xmin)+FONT_BACK_SIZE[0], int(ymin)), (0,0,255,0.3), -1)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255, 0.3), 2)
            cv2.putText(img, obj + '', (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, FONT_SIZE, (255, 255, 255, 255), FONT_THICK)
        cv2.imwrite(os.path.join(det_img_dir, img_name), img)
        print(img_name, ' 已检测完成并保存')

def classify():
  cls_model_name = 'My_Attention_Net_Gated'
  cls_model_weight = 'checkpoint.34.ckpt'
  online_cls_model = Online_Classificaton_Model(cls_model_name, cls_model_weight)
  #cls_f = open('classify.txt', 'a')
  img_list = []
  if not os.path.exists('classify.txt'):
    with open('classify.txt', 'w'):
      pass
  print('实时分类中。。。')
  while True:
    #ori_img_dir = './autotest/ori'   # 存放原始图片的路径
    ori_imgs = os.listdir(ori_img_dir)
    ori_imgs.sort()
    with open('classify.txt', 'r') as cls_f:
      for line in cls_f.readlines():
        img_name = line.strip().split(' ')[0]
        if img_name not in img_list:
          img_list.append(img_name)
    for img_name in ori_imgs:
      if img_name not in img_list:
        if cv2.imread(os.path.join(ori_img_dir, img_name)) is None:
          continue
        results = online_cls_model.predict(os.path.join(ori_img_dir, img_name))
        content = img_name + ' ' + ' '.join(results) + '\n'
        print(img_name + ' 危险品：' + ' '.join(results))
        with open('classify.txt', 'a') as f:
          f.write(content)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='--model 来选择使用的模型')
  parser.add_argument('--model', type=str, help='分类模型：cls')
  arg = parser.parse_args()
  if arg.model == 'cls':
    classify()
  else:
    detect()
  
