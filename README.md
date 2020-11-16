# yolov5_prune
本项目基于[tanluren/yolov3-channel-and-layer-pruning](https://github.com/tanluren/yolov3-channel-and-layer-pruning)实现，将项目扩展到yolov5上。

项目的基本流程是，使用[ultralytics/yolov5](https://github.com/ultralytics/yolov5)训练自己的数据集，在模型性能达到要求但速度未达到要求时，对模型进行剪枝。首先是稀疏化训练，稀疏化训练很重要，如果模型稀疏度不够，剪枝比例过大会导致剪枝后的模型map接近0。剪枝完成后对模型进行微调回复精度。

#### 基础训练
参考yolov5工程

#### 稀疏训练
TODO

#### 通道剪枝策略一
不对shortcut直连的层进行剪枝，避免维度处理。

`python prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune0.pt --percent 0.8`

#### 通道剪枝策略二
对shortcut层也进行了剪枝，剪枝采用每组shortcut中第一个卷积层的mask。
`python shortcut_prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --percent 0.3`

#### 通道剪枝策略三
先以全局阈值找出各卷积层的mask，然后对于每组shortcut，它将相连的各卷积层的剪枝mask取并集，用merge后的mask进行剪枝。
`python slim_prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --global_percent 0.8 --layer_keep 0.01`

#### 八倍通道剪枝
在硬件部署上发现，模型剪枝率相同时，通道数为8的倍数速度最快。（采坑：需要将硬件性能开启到最大）
`python slim_prune_yolov5s_8x.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --global_percent 0.8 --layer_keep 0.01`

#### 层剪枝
针对每一个shortcut层前一个CBL进行评价，对各层的Gmma均值进行排序，取最小的进行层剪枝。为保证yolov5结构完整，这里每剪一个shortcut结构，会同时剪掉一个shortcut层和它前面的两个卷积层。这里只考虑剪主干中的shortcut模块。由于yolov5s也没几个shortcut模块，不打算剪层，所以只实现了该功能，为以后大模型准备。注意，需指定打算剪去shortcut的数量。
`pyhton layer_prune_yolov5s.py --cfg cfg/yolov5s.cfg --data data/fangweisui.data --weights weights/yolov5s_prune1.pt --shortcuts 3`

#### 同时剪层和通道
TODO

#### 微调finetune
TODO




