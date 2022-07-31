### 15th Jul.
开始正式重构一遍之前的pipeline，目前是准备复现yolov6

这次的重构代码，目的是满足未来几年内所有的目标检测相关的模型的实现，
编写时需要慎重并把文档写好。

目前进度在dataset与dataload的编写，主要的annotation格式采用COCO。

### 30th Jul.
data处理部分编写完成，现在进入最关键的loss 和 assignment的编写

1. training部分，搞懂EMA的用法，scheduler的用法，ckpt的相关功能
2. loss 部分，编写lossfunction及assignment method
3. model 部分，目前先编写初步的yolo3，yolox和yolo6？