# yolov5_prune
本项目基于[tanluren/yolov3-channel-and-layer-pruning](https://github.com/tanluren/yolov3-channel-and-layer-pruning)实现，将项目扩展到yolov5上。

项目的基本流程是，使用[ultralytics/yolov5](https://github.com/ultralytics/yolov5)训练自己的数据集，在模型性能达到要求但速度未达到要求时，对模型进行剪枝。首先是稀疏化训练，稀疏化训练很重要，如果模型稀疏度不够，剪枝比例过大会导致剪枝后的模型map接近0。剪枝完成后对模型进行微调回复精度。

## 实例流程
数据集下载[dataset](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz)<br>
### STEP1:基础训练 
附件：[训练记录](https://drive.google.com/drive/folders/1xcFY0nCzGDVu1XzLqjFrsWZbnsBmv9zd?usp=sharing)<br>
### STEP2:稀疏训练     
附件：[稀疏训练记录](https://drive.google.com/drive/folders/13f-SjDT3Hwvnplhi9WZ3NlSaL-aQRjye?usp=sharing)<br>
### STEP3:八倍通道剪枝  
附件：[剪枝后模型](https://drive.google.com/drive/folders/1A2l5wGGShhUgY115l5EmjJoT_6BzkZW_?usp=sharing)<br>
### STEP4:微调finetune 
附件：[微调训练记录](https://drive.google.com/drive/folders/1GIl6hOR5PD3bXVSRyVKziZlo4HiyxKaK?usp=sharing)<br>

## 剪枝步骤
#### STEP1:基础训练
[yolov5第三版](https://github.com/ZJU-lishuang/yolov5-v3) <br>
示例代码<br>
```
python train.py --img 640 --batch 16 --epochs 100 --data ./data/hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/yolov5s.pt --name s_hand
```

#### STEP2:稀疏训练
--prune 0 适用于通道剪枝策略一，--prune 1 适用于其他剪枝策略。<br>
[yolov5第三版](https://github.com/ZJU-lishuang/yolov5-v3)<br>
示例代码<br>
```
python train_sparsity.py --img 640 --batch 16 --epochs 300 --data data/hand.yaml --cfg models/yolov5s_hand.yaml --weights runs/train/s_hand100/weights/last.pt --name s_to_prune -sr --s 0.001 --prune 1
```

#### STEP3:通道剪枝策略一
不对shortcut直连的层进行剪枝，避免维度处理。<br>
`python prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune0.pt --percent 0.8`

#### STEP3:通道剪枝策略二
对shortcut层也进行了剪枝，剪枝采用每组shortcut中第一个卷积层的mask。<br>
`python shortcut_prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --percent 0.3`

#### STEP3:通道剪枝策略三
先以全局阈值找出各卷积层的mask，然后对于每组shortcut，它将相连的各卷积层的剪枝mask取并集，用merge后的mask进行剪枝。<br>
`python slim_prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --global_percent 0.8 --layer_keep 0.01`

#### STEP3:八倍通道剪枝
在硬件部署上发现，模型剪枝率相同时，通道数为8的倍数速度最快。（采坑：需要将硬件性能开启到最大）<br>
示例代码<br>
```
python slim_prune_yolov5s_8x.py --cfg cfg/yolov5s_v3_hand.cfg --data data/oxfordhand.data --weights weights/last_s_v3_to_prune.pt --global_percent 0.1 --layer_keep 0.01
```

#### STEP4:微调finetune
[yolov5第三版](https://github.com/ZJU-lishuang/yolov5-v3)<br>
```
python prune_finetune.py --img 640 --batch 16 --epochs 100 --data ./data/hand.yaml --cfg ./cfg/prune_0.8_keep_0.01_8x_yolov5s_hand.cfg --weights ./weights/prune_0.8_keep_0.01_8x_last_s_to_prune.pt --name prune_hand_s
```




