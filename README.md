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

5. 运行
`
python deploy.py
`


