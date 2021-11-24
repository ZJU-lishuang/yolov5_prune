import cv2
import torch
import random
import argparse
import numpy as np
# from utils.modelscfg import Darknet
# from utils.augmentations import letterbox
# from utils.general import non_max_suppression, scale_coords


def copy_conv(conv_src,conv_dst):
    conv_dst[0] = conv_src.conv
    conv_dst[1] = conv_src.bn
    conv_dst[2] = conv_src.act

def copy_conv_idx(conv_src,conv_dst,idx):
    copy_conv(conv_src,conv_dst)
    return idx+1

def copy_c3(c3_src,module_lists,idx,num,shortcut=True):
    # C3-cv2
    idx=copy_conv_idx(c3_src.cv2, module_lists[idx],idx)
    # route
    idx = idx + 1
    # C3-cv1
    idx=copy_conv_idx(c3_src.cv1, module_lists[idx],idx)
    # C3-m
    for i in range(num):
        # Bottleneck-cv1
        idx=copy_conv_idx(c3_src.m[i].cv1, module_lists[idx],idx)
        # Bottleneck-cv2
        idx=copy_conv_idx(c3_src.m[i].cv2, module_lists[idx],idx)
        if shortcut:
            #m-add
            idx = idx + 1
    #C3-cat
    idx = idx + 1
    #C3--cv3
    idx=copy_conv_idx(c3_src.cv3, module_lists[idx],idx)
    return idx

def copy_weight_v6(modelyolov5,model):
    idx=0
    depth_multiple=1
    if 'depth_multiple' in modelyolov5.yaml:
        depth_multiple=modelyolov5.yaml['depth_multiple']
    conv0 = list(modelyolov5.model.children())[0]
    idx=copy_conv_idx(conv0, model.module_list[idx],idx)
    conv1 = list(modelyolov5.model.children())[1]
    idx=copy_conv_idx(conv1, model.module_list[idx],idx)
    cspnet2 = list(modelyolov5.model.children())[2]
    idx=copy_c3(cspnet2,model.module_list,idx, round(3 * depth_multiple))
    conv3 = list(modelyolov5.model.children())[3]
    idx=copy_conv_idx(conv3, model.module_list[idx],idx)
    cspnet4 = list(modelyolov5.model.children())[4]
    idx = copy_c3(cspnet4, model.module_list,idx, round(6 * depth_multiple))
    conv5 = list(modelyolov5.model.children())[5]
    idx=copy_conv_idx(conv5, model.module_list[idx],idx)
    cspnet6 = list(modelyolov5.model.children())[6]
    idx = copy_c3(cspnet6, model.module_list,idx, round(9 * depth_multiple))
    conv7 = list(modelyolov5.model.children())[7]
    idx=copy_conv_idx(conv7, model.module_list[idx],idx)
    cspnet8 = list(modelyolov5.model.children())[8]
    idx = copy_c3(cspnet8, model.module_list,idx, round(3 * depth_multiple))
    sppf9 = list(modelyolov5.model.children())[9]
    idx=copy_conv_idx(sppf9.cv1, model.module_list[idx],idx)
    model.module_list[idx] = sppf9.m
    idx = idx + 1
    model.module_list[idx] = sppf9.m
    idx = idx + 1
    model.module_list[idx] = sppf9.m
    idx = idx + 1
    #route
    idx = idx + 1
    idx=copy_conv_idx(sppf9.cv2, model.module_list[idx],idx)
    conv10 = list(modelyolov5.model.children())[10]
    idx=copy_conv_idx(conv10, model.module_list[idx],idx)
    upsample11 = list(modelyolov5.model.children())[11]
    model.module_list[idx] = upsample11
    idx = idx + 1
    #route
    idx=idx+1
    cspnet13 = list(modelyolov5.model.children())[13]
    idx = copy_c3(cspnet13, model.module_list, idx,round(3 * depth_multiple),False)
    conv14 = list(modelyolov5.model.children())[14]
    idx=copy_conv_idx(conv14, model.module_list[idx],idx)
    upsample15 = list(modelyolov5.model.children())[15]
    model.module_list[idx] = upsample15
    idx = idx + 1
    # route
    idx = idx + 1
    cspnet17 = list(modelyolov5.model.children())[17]
    idx = copy_c3(cspnet17, model.module_list, idx,round(3 * depth_multiple), False)
    #conv
    conv_detect1_idx=idx
    idx=idx+1
    #yolo
    idx=idx+1
    #route
    idx=idx+1
    conv18 = list(modelyolov5.model.children())[18]
    idx=copy_conv_idx(conv18, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet20 = list(modelyolov5.model.children())[20]
    idx = copy_c3(cspnet20, model.module_list,idx, round(3 * depth_multiple), False)
    # conv
    conv_detect2_idx = idx
    idx = idx + 1
    # yolo
    idx = idx + 1
    # route
    idx = idx + 1
    conv21 = list(modelyolov5.model.children())[21]
    idx=copy_conv_idx(conv21, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet23 = list(modelyolov5.model.children())[23]
    idx = copy_c3(cspnet23, model.module_list, idx,round(3 * depth_multiple), False)
    # conv
    conv_detect3_idx = idx
    detect24 = list(modelyolov5.model.children())[24]
    model.module_list[conv_detect1_idx][0] = detect24.m[0]
    model.module_list[conv_detect2_idx][0] = detect24.m[1]
    model.module_list[conv_detect3_idx][0] = detect24.m[2]

# def plot_one_box(x, img, color=None, label=None, line_thickness=None):
#     # Plots one bounding box on image img
#     tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)  # font thickness
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
#
# def img_det(model,img,img0,save_path):
#     model.eval()
#     pred = model(img)[0]
#
#     pred = non_max_suppression(pred, 0.4, 0.5, classes=None,
#                                agnostic=False)
#     # Process detections
#     for i, det in enumerate(pred):  # detections per image
#         if det is not None and len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
#
#             # Write results
#             for *xyxy, conf, cls in det:
#                 label = '%s %.2f' % (str(int(cls)), conf)
#                 plot_one_box(xyxy, img0, label=label, color=[random.randint(0, 255) for _ in range(3)],
#                              line_thickness=3)
#             cv2.imwrite(save_path, img0)
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='../cfg/yolov5s_v6.cfg', help='cfg file path')
#     parser.add_argument('--path', type=str, default='../data/images/bus.jpg', help='img file path')
#     parser.add_argument('--weights', type=str, default='../weights/yolov5s.pt', help='sparse model weights')
#     parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
#     opt = parser.parse_args()
#     print(opt)
#
#     img_size = opt.img_size
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # loading yolov5s
#     modelyolov5 = torch.load(opt.weights, map_location=device)['model'].float().eval()
#
#     from models.yolo import Detect
#     inplace = True
#     for m in modelyolov5.modules():
#         if type(m) is Detect:
#             m.inplace = inplace
#             if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
#                 delattr(m, 'anchor_grid')
#                 setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
#
#     # load yolov5s from cfg
#     model = Darknet(opt.cfg, (img_size, img_size)).to(device)
#     copy_weight_v6(modelyolov5, model)
#
#     path = opt.path
#     img0 = cv2.imread(path)  # BGR
#     # Padded resize
#     img = letterbox(img0, new_shape=416)[0]
#
#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).to(device)
#     img = img.float()
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#
#     save_path="../v5_cfg.jpg"
#     img_det(model,img, img0, save_path)
#     yolov5save_path = "../v5.jpg"
#     img_det(modelyolov5, img,img0, yolov5save_path)