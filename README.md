# yolov5_prune
本项目基于[tanluren/yolov3-channel-and-layer-pruning](https://github.com/tanluren/yolov3-channel-and-layer-pruning)实现，将项目扩展到yolov5上。

项目的基本流程是，使用[ultralytics/yolov5](https://github.com/ultralytics/yolov5)训练自己的数据集，在模型性能达到要求但速度未达到要求时，对模型进行剪枝。首先是稀疏化训练，稀疏化训练很重要，如果模型稀疏度不够，剪枝比例过大会导致剪枝后的模型map接近0。剪枝完成后对模型进行微调回复精度。

本项目使用的yolov5为第四版本。

TODO: 已完成蒸馏实验，蒸馏在微调模型上作用明显，近期更新相关代码和步骤。

## 实例流程
数据集下载[dataset](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz)<br>
### STEP1:基础训练 
附件：[训练记录](https://drive.google.com/drive/folders/1v0HZYBhU6d4M2hvEfjia76wYbQlaFz_f?usp=sharing)<br>
### STEP2:稀疏训练     
附件：[稀疏训练记录](https://drive.google.com/drive/folders/1tJaeSOzQlyrx1l22hhop8G3ZuKshm8rp?usp=sharing)<br>
### STEP3:八倍通道剪枝  
附件：[剪枝后模型](https://drive.google.com/drive/folders/1V5nA6oGXX43bagpO3cJIFpI0zjAOzt0p?usp=sharing)<br>
### STEP4:微调finetune 
附件：[微调训练记录](https://drive.google.com/drive/folders/1vT_pN_XlMBniF9YXaPj2KeCNZitxYFLA?usp=sharing)<br>

## 剪枝步骤
#### STEP1:基础训练
[yolov5](https://github.com/ZJU-lishuang/yolov5-v4) <br>
示例代码 <br>
```
python train.py --img 640 --batch 8 --epochs 50 --weights weights/yolov5s_v4.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --name s_hand
```

#### STEP2:稀疏训练
--prune 0 适用于通道剪枝策略一，--prune 1 适用于其他剪枝策略。<br>
[yolov5](https://github.com/ZJU-lishuang/yolov5-v4)<br>
示例代码<br>
```
python train_sparsity.py --img 640 --batch 8 --epochs 50 --data data/coco_hand.yaml --cfg models/yolov5s.yaml --weights runs/train/s_hand/weights/last.pt --name s_hand_sparsity -sr --s 0.001 --prune 1
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
示例代码<br>
```
python slim_prune_yolov5s_8x.py --cfg cfg/yolov5s_v4_hand.cfg --data data/oxfordhand.data --weights weights/last_v4s.pt --global_percent 0.5 --layer_keep 0.01 --img_size 640
```

#### STEP4:微调finetune
[yolov5](https://github.com/ZJU-lishuang/yolov5-v4)<br>
示例代码<br>
```
python prune_finetune.py --img 640 --batch 8 --epochs 50 --data data/coco_hand.yaml --cfg ./cfg/prune_0.5_keep_0.01_8x_yolov5s_v4_hand.cfg --weights ./weights/prune_0.5_keep_0.01_8x_last_v4s.pt --name s_hand_finetune
```

#### STEP5:剪枝后模型推理
[yolov5](https://github.com/ZJU-lishuang/yolov5-v4)<br>
示例代码<br>
```shell
python prune_detect.py --weights weights/last_s_hand_finetune.pt --img  640 --conf 0.7 --save-txt --source inference/images
```


