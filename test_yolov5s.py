from modelsori import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse

from models.yolo import Model

import torchvision

def letterboxv5(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        # dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppressionv5(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def copy_conv(conv_src,conv_dst):
    conv_dst[0] = conv_src.conv
    conv_dst[1] = conv_src.bn
    conv_dst[2] = conv_src.act

def copy_weight_v4(modelyolov5,model):
    focus = list(modelyolov5.model.children())[0]
    copy_conv(focus.conv, model.module_list[1])
    conv1 = list(modelyolov5.model.children())[1]
    copy_conv(conv1, model.module_list[2])
    cspnet1 = list(modelyolov5.model.children())[2]
    copy_conv(cspnet1.cv2, model.module_list[3])
    copy_conv(cspnet1.cv1, model.module_list[5])
    copy_conv(cspnet1.m[0].cv1, model.module_list[6])
    copy_conv(cspnet1.m[0].cv2, model.module_list[7])
    copy_conv(cspnet1.cv3, model.module_list[10])
    conv2 = list(modelyolov5.model.children())[3]
    copy_conv(conv2, model.module_list[11])
    cspnet2 = list(modelyolov5.model.children())[4]
    copy_conv(cspnet2.cv2, model.module_list[12])
    copy_conv(cspnet2.cv1, model.module_list[14])
    copy_conv(cspnet2.m[0].cv1, model.module_list[15])
    copy_conv(cspnet2.m[0].cv2, model.module_list[16])
    copy_conv(cspnet2.m[1].cv1, model.module_list[18])
    copy_conv(cspnet2.m[1].cv2, model.module_list[19])
    copy_conv(cspnet2.m[2].cv1, model.module_list[21])
    copy_conv(cspnet2.m[2].cv2, model.module_list[22])
    copy_conv(cspnet2.cv3, model.module_list[25])
    conv3 = list(modelyolov5.model.children())[5]
    copy_conv(conv3, model.module_list[26])
    cspnet3 = list(modelyolov5.model.children())[6]
    copy_conv(cspnet3.cv2, model.module_list[27])
    copy_conv(cspnet3.cv1, model.module_list[29])
    copy_conv(cspnet3.m[0].cv1, model.module_list[30])
    copy_conv(cspnet3.m[0].cv2, model.module_list[31])
    copy_conv(cspnet3.m[1].cv1, model.module_list[33])
    copy_conv(cspnet3.m[1].cv2, model.module_list[34])
    copy_conv(cspnet3.m[2].cv1, model.module_list[36])
    copy_conv(cspnet3.m[2].cv2, model.module_list[37])
    copy_conv(cspnet3.cv3, model.module_list[40])
    conv4 = list(modelyolov5.model.children())[7]
    copy_conv(conv4, model.module_list[41])
    spp = list(modelyolov5.model.children())[8]
    copy_conv(spp.cv1, model.module_list[42])
    model.module_list[43] = spp.m[0]
    model.module_list[45] = spp.m[1]
    model.module_list[47] = spp.m[2]
    copy_conv(spp.cv2, model.module_list[49])
    cspnet4 = list(modelyolov5.model.children())[9]
    copy_conv(cspnet4.cv2, model.module_list[50])
    copy_conv(cspnet4.cv1, model.module_list[52])
    copy_conv(cspnet4.m[0].cv1, model.module_list[53])
    copy_conv(cspnet4.m[0].cv2, model.module_list[54])
    copy_conv(cspnet4.cv3, model.module_list[56])
    conv5 = list(modelyolov5.model.children())[10]
    copy_conv(conv5, model.module_list[57])
    upsample1 = list(modelyolov5.model.children())[11]
    model.module_list[58] = upsample1
    cspnet5 = list(modelyolov5.model.children())[13]
    copy_conv(cspnet5.cv2, model.module_list[60])
    copy_conv(cspnet5.cv1, model.module_list[62])
    copy_conv(cspnet5.m[0].cv1, model.module_list[63])
    copy_conv(cspnet5.m[0].cv2, model.module_list[64])
    copy_conv(cspnet5.cv3, model.module_list[66])
    conv6 = list(modelyolov5.model.children())[14]
    copy_conv(conv6, model.module_list[67])
    upsample2 = list(modelyolov5.model.children())[15]
    model.module_list[68] = upsample2
    cspnet6 = list(modelyolov5.model.children())[17]
    copy_conv(cspnet6.cv2, model.module_list[70])
    copy_conv(cspnet6.cv1, model.module_list[72])
    copy_conv(cspnet6.m[0].cv1, model.module_list[73])
    copy_conv(cspnet6.m[0].cv2, model.module_list[74])
    copy_conv(cspnet6.cv3, model.module_list[76])
    conv7 = list(modelyolov5.model.children())[18]
    copy_conv(conv7, model.module_list[80])
    cspnet7 = list(modelyolov5.model.children())[20]
    copy_conv(cspnet7.cv2, model.module_list[82])
    copy_conv(cspnet7.cv1, model.module_list[84])
    copy_conv(cspnet7.m[0].cv1, model.module_list[85])
    copy_conv(cspnet7.m[0].cv2, model.module_list[86])
    copy_conv(cspnet7.cv3, model.module_list[88])
    conv8 = list(modelyolov5.model.children())[21]
    copy_conv(conv8, model.module_list[92])
    cspnet8 = list(modelyolov5.model.children())[23]
    copy_conv(cspnet8.cv2, model.module_list[94])
    copy_conv(cspnet8.cv1, model.module_list[96])
    copy_conv(cspnet8.m[0].cv1, model.module_list[97])
    copy_conv(cspnet8.m[0].cv2, model.module_list[98])
    copy_conv(cspnet8.cv3, model.module_list[100])
    detect = list(modelyolov5.model.children())[24]
    model.module_list[77][0] = detect.m[0]
    model.module_list[89][0] = detect.m[1]
    model.module_list[101][0] = detect.m[2]

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov5s_v4.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov5s_v4.pt', help='sparse model weights')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #the way of loading yolov5s
    # ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    # modelyolov5 = Model('models/yolov5s_v4.yaml', nc=80).to(device)
    # exclude = ['anchor']  # exclude keys
    # ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
    #                  if k in modelyolov5.state_dict() and not any(x in k for x in exclude)
    #                  and modelyolov5.state_dict()[k].shape == v.shape}
    # modelyolov5.load_state_dict(ckpt['model'], strict=False)

    #another way of loading yolov5s
    modelyolov5=torch.load(opt.weights, map_location=device)['model'].float().eval()
    modelyolov5.model[24].export = False  # onnx export

    # model=modelyolov5

    #load yolov5s from cfg
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)
    copy_weight_v4(modelyolov5,model)

    path='data/samples/bus.jpg'
    img0 = cv2.imread(path)  # BGR
    # Padded resize
    img = letterboxv5(img0, new_shape=416)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # modelyolov5.eval()


    model.eval()
    pred = model(img)[0]

    pred = non_max_suppressionv5(pred, 0.4, 0.5, classes=None,
                               agnostic=False)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (str(int(cls)), conf)
                plot_one_box(xyxy, img0, label=label, color=[random.randint(0, 255) for _ in range(3)], line_thickness=3)
            cv2.imwrite("v5_cfg.jpg", img0)

    modelyolov5.eval()
    pred = modelyolov5(img)[0]

    pred = non_max_suppressionv5(pred, 0.4, 0.5, classes=None,
                                 agnostic=False)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (str(int(cls)), conf)
                plot_one_box(xyxy, img0, label=label, color=[random.randint(0, 255) for _ in range(3)],
                             line_thickness=3)
            cv2.imwrite("v5.jpg", img0)
