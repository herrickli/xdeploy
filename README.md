### Requirements
- cuda 9.0
- pytorch 1.1

### conda安装pytorch
1. 切换清华源, 打开 ~/.condarc, 删除所有内容，输入以下内容
```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
show_channel_urls: true
```
2. 创建conda虚拟环境并安装pytorch相关库
```
conda create -n deploy python=3.6
conda activate deploy
conda install pytorch=1.1 torchvision cudatoolkit=9.0 cudnn
````
3. 安装依赖
```
pip install -r requirements.txt
```

4. 编译
```
pip install -v -e .
```

5. 运行（网页检测使用）
`
python deploy.py
`

## 自动读取图片
### 使用检测模型自动读取图片
```
python autoread.py
```
- 检测模型将未检测的图片和检测后的图片分别存在两个文件夹中，请打开`autoread.py`修改这两个文件夹

### 使用分类模型自动读取图片
```
phthon autoread.py --model cls
```
- 分类模型不会输出可视化的框，它的分类结果直接显示在界面上。
- 可以打开`classify.txt`文件查看每张图片的分类结果
