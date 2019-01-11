# Raspberry Pi CM3使用Movidius
本项目在ubuntu X64系统下使用https://github.com/thtrieu/darkflow 训练和转换模型，在树莓派下使用Movidius转换格式并测试使用。
## 克隆我的开源项目
git clone https://github.com/17702513221/Movidius.git
## 将darkflow训练的yolov2-tiny-voc.pb文件复制到model文件夹下，模型格式转换
'cd Movidius/ncs_objection/model'
'mvNCCompile -o tiny_yolo_v2.graph yolov2-tiny-voc.pb -s 12'
## 测试模型
'cd ncs_objection'
### 默认为测试test文件夹下图片，置信度0.3
'python3 test.py'
### 测试视频（默认为测试test文件夹下视频，置信度0.3）
'python3 test.py -m video'
### 使用摄像头
'python3 test.py -m video -v 0'
### 修改置信度0.1(修改图片，视频地址不列出-i -v)
'python3 test.py -c 0.1'
### 使用树莓派摄像头
'cd /etc/modules-load.d'
'sudo nano modules.conf'
添加：'bcm2835-v4l2'
'reboot'
### 目前效果:神经计算棒2代识别一次要0.25秒，更新0.3秒一帧
'python3 video_objects_scalable_yolov2_tiny.py'
