from mmdet.apis import init_detector, inference_detector
import mmcv
from flask import Flask, request, Blueprint, render_template, flash
from werkzeug.utils import secure_filename
import base64
import os
import cv2

config_file = './configs/config.py'
weight_file = './weights/weight.pth'

model = init_detector(config_file, weight_file, device='cuda:0')

class_names = ['hammar', 'scissors', 'knife', 'bottle', 'battery', 'firecracker', 'gun', 'grenade', 'bullet', 'lighter', 'ppball', 'baton']

#results = inference_detector(model, 'cur.jpg')

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

#img = 'demo.jpg'
#result = inference_detector(model, img)

#print('result is generated!')

#img = 'demo.jpg'
#result = inference_detector(model, img)

#print('result is generated!')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Lewd did I live, and evil I did dwel.'
@app.route('/', methods=['POST', 'GET'])
def detect():
    if request.method == 'POST':
        request_data = request.form
        #name = request_data['name']
        img = request_data['content']
        img = base64.b64decode(img)
        with open('cur.jpg', 'wb') as f :
            f.write(img)
        result = inference_detector(model, 'cur.jpg')
        result = parse_result(result)
        return '{}'.format(result)
    return 'GET: result!'

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', name = 'static/smail.jpg')
    elif request.method == 'POST':
        ori_img = request.files['xray-img']
        name = secure_filename(ori_img.filename)
        if not name.endswith(('jpg', 'png', 'jpeg', 'JPEG')):
            flash('只支持 jpg png jpeg JPEG 格式文件！')
            print(name)
            return render_template('index.html', name = 'static/smail.jpg')
        ori_img.save('static/imgs_ori/' + name)
        results = inference_detector(model, 'static/imgs_ori/' + name)
        results = parse_result(results)
        img = cv2.imread('static/imgs_ori/'+name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        for result in results:
            obj, score, box = result
            if score > 0.5:
                xmin, ymin, xmax, ymax = box
                cv2.putText(img, obj + ' ' + str(score), (int(xmin), int(ymin)+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255, 255), 1)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255, 255), 1)
        cv2.imwrite('static/imgs_detect/'+name, img)
        return render_template('index.html', name='static/imgs_detect/'+name)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
