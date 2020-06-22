from flask_cors import CORS
import argparse
import mmcv
from flask import Flask, request, Blueprint, render_template, flash, Response
from werkzeug.utils import secure_filename
import base64
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy
 
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False)

# default model from cheng
# detection model
from mmdet.apis import init_detector
from mmdet.apis import inference_detector as predict
config_file = './configs/config.py'
weight_file = './weights/weight.pth'
model = init_detector(config_file, weight_file, device='cuda:0')
# classify model
from online_model import Online_Classificaton_Model
cls_model_name = 'My_Attention_Net_Gated'
cls_model_weight = 'checkpoint.34.ckpt'
img_path = '/home/licheng/projects/mmdetection/data/41test-images/IMG_20191210_143726898_0.JPEG'
online_cls_model = Online_Classificaton_Model(cls_model_name, cls_model_weight)
#predictions = online_cls_model.predict(img_path)
#print(predictions)

class_names = ['hammar', 'scissors', 'knife', 'bottle', 'battery', 'firecracker', 'gun', 'grenade', 'bullet', 'lighter', 'ppball', 'baton']
class_chinese = ['锤', '剪刀', '刀', '瓶子', '电池', '烟花', '枪', '手雷', '子弹', '打火机', '乒乓球', '甩棍']
#results = inference_detector(model, 'cur.jpg')


def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "fonts/simhei.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


def parse_result(results):
    boxes = []
    for class_index, result in enumerate(results):
        if len(result) == 0:
            continue
        class_name = class_chinese[class_index]
        for bounding_box in result:
            xmin, ymin, xmax, ymax, score = bounding_box
            boxes.append([class_name, score, [xmin, ymin, xmax, ymax]])
    return boxes

app = Flask(__name__, template_folder='./', static_folder='/', static_url_path='/')
print(__name__)
CORS(app, supports_credential=True)
app.config['SECRET_KEY'] = 'Lewd did I live, and evil I did dwel.'

FONT_FACE  = cv2.FONT_HERSHEY_COMPLEX
FONT_SIZE  = 1
FONT_THICK = 1

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html', name = 'static/smail.jpg')
    elif request.method == 'POST':
        obj = request.files['file']
        request_data = request.form
        #print(request_data['type'])
        name = secure_filename(obj.filename)
        ori_img_path = 'static/imgs_ori/' + name
        obj.save(ori_img_path)
        if request_data['type'] == 'detect':
            results = predict(model, ori_img_path)
            results = parse_result(results)
            img = cv2.imread('static/imgs_ori/'+name)
            for result in results:
                obj, score, box = result
                FONT_BACK_SIZE = cv2.getTextSize(obj, FONT_FACE, FONT_SIZE, FONT_THICK)[0]
                if score > 0.5:
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(img, (int(xmin), int(ymin)-FONT_BACK_SIZE[1]), (int(xmin)+FONT_BACK_SIZE[0], int(ymin)), (0,0,255,0.3), -1)
                    #cv2.putText(img, obj + '', (int(xmin), int(ymin)), FONT_FACE, FONT_SIZE, (255, 255, 255, 0.7), FONT_THICK)
                    img = cv2ImgAddText(img, obj, int(xmin), int(ymin)-FONT_BACK_SIZE[1], (255,255,255), 20)
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255, 0.3), 2)
            cv2.imwrite('static/imgs_detect/'+name, img)
            #with open('static/imgs_detect/'+name, 'rb') as f:
            #    pred = f.read()
            #resp = Response(pred, mimetype='image/jpeg')
            pred_path = 'static/imgs_detect/' + name
            return pred_path
        elif request_data['type'] == 'classify':
            predictions = online_cls_model.predict('static/imgs_ori/'+name)
            content = ','.join(predictions)
            return content
            #return 'hammer, scissors, knife, bottle, battery, firecracker, gun, grenade, bullet, lighter, ppball, baton'
    return 'error'

@app.route('/zm', methods=['GET'])
def zm():
    return render_template('purpleCode.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
