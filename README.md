# yolov5_prune
本项目基于[tanluren/yolov3-channel-and-layer-pruning](https://github.com/tanluren/yolov3-channel-and-layer-pruning)实现，将项目扩展到yolov5上。

项目的基本流程是，使用[ultralytics/yolov5](https://github.com/ultralytics/yolov5)训练自己的数据集，在模型性能达到要求但速度未达到要求时，对模型进行剪枝。首先是稀疏化训练，稀疏化训练很重要，如果模型稀疏度不够，剪枝比例过大会导致剪枝后的模型map接近0。剪枝完成后对模型进行微调回复精度。

[yolov5第三版本](README_v3.md)

## 实例流程
数据集下载[dataset](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz)<br>
### STEP1:基础训练 
附件：[训练记录](https://drive.google.com/drive/folders/1xHq4m-X5vrrCtIajyMFTO8ClZlxJOjD_?usp=sharing)<br>
注：该模型使用torch1.4训练，后面三步环境torch>1.5，加载该模型时需添加
```
if weights.endswith('.pt'):  # pytorch format
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        #低版本模型在高版本中加载
        for k, m in ckpt['model'].named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
```
### STEP2:稀疏训练     
附件：[稀疏训练记录](https://drive.google.com/drive/folders/1XTkS_aTzc9MEGZVtLxMISE2WLIKRv4hT?usp=sharing)<br>
### STEP3:八倍通道剪枝  
附件：[剪枝后模型](https://drive.google.com/drive/folders/1_SPlU2nmy5-TDfL0JsfZqwZxK6pI_Sco?usp=sharing)<br>
### STEP4:微调finetune 
附件：[微调训练记录](https://drive.google.com/drive/folders/1tDPUGEzCPil5mL1MNS_2IY8knqBvaAGu?usp=sharing)<br>

## 剪枝步骤
#### STEP1:基础训练
[yolov5第二版](https://github.com/ZJU-lishuang/yolov5) <br>
示例代码 <br>
```
python train_pytorch1.4_noprune.py --img 640 --batch 8 --epochs 100 --data ./data/hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/yolov5s.pt --name s_hand
```

#### STEP2:稀疏训练
--prune 0 适用于通道剪枝策略一，--prune 1 适用于其他剪枝策略。<br>
[yolov5第二版](https://github.com/ZJU-lishuang/yolov5)<br>
示例代码<br>
```
python train_pytorch1.4.py --img 640 --batch 16 --epochs 300 --data ./data/hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/last_s_hand.pt --name s_to_prune -sr --s 0.001 --prune 1
```

#### STEP3:通道剪枝策略一
不对shortcut直连的层进行剪枝，避免维度处理。<br>
```
python prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune0.pt --percent 0.8
```

#### STEP3:通道剪枝策略二
对shortcut层也进行了剪枝，剪枝采用每组shortcut中第一个卷积层的mask。<br>
```
python shortcut_prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --percent 0.3
```

#### STEP3:通道剪枝策略三
先以全局阈值找出各卷积层的mask，然后对于每组shortcut，它将相连的各卷积层的剪枝mask取并集，用merge后的mask进行剪枝。<br>
```
python slim_prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --global_percent 0.8 --layer_keep 0.01
```

#### STEP3:八倍通道剪枝
在硬件部署上发现，模型剪枝率相同时，通道数为8的倍数速度最快。（采坑：需要将硬件性能开启到最大）<br>
```
python slim_prune_yolov5s_8x.py --cfg cfg/yolov5s_hand.cfg --data data/oxfordhand.data --weights weights/last_s_to_prune.pt --global_percent 0.8 --layer_keep 0.01
```

#### STEP3:层剪枝
针对每一个shortcut层前一个CBL进行评价，对各层的Gmma均值进行排序，取最小的进行层剪枝。为保证yolov5结构完整，这里每剪一个shortcut结构，会同时剪掉一个shortcut层和它前面的两个卷积层。这里只考虑剪主干中的shortcut模块。由于yolov5s也没几个shortcut模块，不打算剪层，所以只实现了该功能，为以后大模型准备。注意，需指定打算剪去shortcut的数量。<br>
```
pyhton layer_prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --shortcuts 3
```

#### STEP3:同时剪层和通道
TODO

#### STEP4:微调finetune
[yolov5第二版](https://github.com/ZJU-lishuang/yolov5)<br>
示例代码<br>
```
python prune_finetune.py --img 640 --batch 8 --epochs 10 --data ./data/hand.yaml --cfg ./cfg/prune_0.8_keep_0.01_8x_yolov5s_hand.cfg --weights ./weights/prune_0.8_keep_0.01_8x_last_s_to_prune.pt --name prune_hand_s
```

#### TODO
* layer_prune_yolov5s.py,prune_yolov5s.py,shortcut_prune_yolov5s.py中，对convolutional_noconv剪枝导致通道数为0的bug fix。


