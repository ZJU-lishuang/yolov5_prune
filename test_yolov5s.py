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

def copy_weight(modelyolov5,model):
    focus = list(modelyolov5.model.children())[0]
    model.module_list[1][0] = focus.conv.conv
    model.module_list[1][1] = focus.conv.bn
    model.module_list[1][2] = focus.conv.act
    conv1 = list(modelyolov5.model.children())[1]
    model.module_list[2][0] = conv1.conv
    model.module_list[2][1] = conv1.bn
    model.module_list[2][2] = conv1.act
    cspnet1 = list(modelyolov5.model.children())[2]
    model.module_list[3][0] = cspnet1.cv2
    model.module_list[5][0] = cspnet1.cv1.conv
    model.module_list[5][1] = cspnet1.cv1.bn
    model.module_list[5][2] = cspnet1.cv1.act
    model.module_list[9][0] = cspnet1.cv3
    model.module_list[11][0] = cspnet1.bn
    model.module_list[11][1] = cspnet1.act
    model.module_list[6][0] = cspnet1.m[0].cv1.conv
    model.module_list[6][1] = cspnet1.m[0].cv1.bn
    model.module_list[6][2] = cspnet1.m[0].cv1.act
    model.module_list[7][0] = cspnet1.m[0].cv2.conv
    model.module_list[7][1] = cspnet1.m[0].cv2.bn
    model.module_list[7][2] = cspnet1.m[0].cv2.act
    model.module_list[12][0] = cspnet1.cv4.conv
    model.module_list[12][1] = cspnet1.cv4.bn
    model.module_list[12][2] = cspnet1.cv4.act
    conv2 = list(modelyolov5.model.children())[3]
    model.module_list[13][0] = conv2.conv
    model.module_list[13][1] = conv2.bn
    model.module_list[13][2] = conv2.act
    cspnet2 = list(modelyolov5.model.children())[4]
    model.module_list[14][0] = cspnet2.cv2
    model.module_list[16][0] = cspnet2.cv1.conv
    model.module_list[16][1] = cspnet2.cv1.bn
    model.module_list[16][2] = cspnet2.cv1.act
    model.module_list[26][0] = cspnet2.cv3
    model.module_list[28][0] = cspnet2.bn
    model.module_list[28][1] = cspnet2.act
    model.module_list[29][0] = cspnet2.cv4.conv
    model.module_list[29][1] = cspnet2.cv4.bn
    model.module_list[29][2] = cspnet2.cv4.act
    model.module_list[17][0] = cspnet2.m[0].cv1.conv
    model.module_list[17][1] = cspnet2.m[0].cv1.bn
    model.module_list[17][2] = cspnet2.m[0].cv1.act
    model.module_list[18][0] = cspnet2.m[0].cv2.conv
    model.module_list[18][1] = cspnet2.m[0].cv2.bn
    model.module_list[18][2] = cspnet2.m[0].cv2.act
    model.module_list[20][0] = cspnet2.m[1].cv1.conv
    model.module_list[20][1] = cspnet2.m[1].cv1.bn
    model.module_list[20][2] = cspnet2.m[1].cv1.act
    model.module_list[21][0] = cspnet2.m[1].cv2.conv
    model.module_list[21][1] = cspnet2.m[1].cv2.bn
    model.module_list[21][2] = cspnet2.m[1].cv2.act
    model.module_list[23][0] = cspnet2.m[2].cv1.conv
    model.module_list[23][1] = cspnet2.m[2].cv1.bn
    model.module_list[23][2] = cspnet2.m[2].cv1.act
    model.module_list[24][0] = cspnet2.m[2].cv2.conv
    model.module_list[24][1] = cspnet2.m[2].cv2.bn
    model.module_list[24][2] = cspnet2.m[2].cv2.act
    conv3 = list(modelyolov5.model.children())[5]
    model.module_list[30][0] = conv3.conv
    model.module_list[30][1] = conv3.bn
    model.module_list[30][2] = conv3.act
    cspnet3 = list(modelyolov5.model.children())[6]
    model.module_list[31][0] = cspnet3.cv2
    model.module_list[33][0] = cspnet3.cv1.conv
    model.module_list[33][1] = cspnet3.cv1.bn
    model.module_list[33][2] = cspnet3.cv1.act
    model.module_list[43][0] = cspnet3.cv3
    model.module_list[45][0] = cspnet3.bn
    model.module_list[45][1] = cspnet3.act
    model.module_list[46][0] = cspnet3.cv4.conv
    model.module_list[46][1] = cspnet3.cv4.bn
    model.module_list[46][2] = cspnet3.cv4.act
    model.module_list[34][0] = cspnet3.m[0].cv1.conv
    model.module_list[34][1] = cspnet3.m[0].cv1.bn
    model.module_list[34][2] = cspnet3.m[0].cv1.act
    model.module_list[35][0] = cspnet3.m[0].cv2.conv
    model.module_list[35][1] = cspnet3.m[0].cv2.bn
    model.module_list[35][2] = cspnet3.m[0].cv2.act
    model.module_list[37][0] = cspnet3.m[1].cv1.conv
    model.module_list[37][1] = cspnet3.m[1].cv1.bn
    model.module_list[37][2] = cspnet3.m[1].cv1.act
    model.module_list[38][0] = cspnet3.m[1].cv2.conv
    model.module_list[38][1] = cspnet3.m[1].cv2.bn
    model.module_list[38][2] = cspnet3.m[1].cv2.act
    model.module_list[40][0] = cspnet3.m[2].cv1.conv
    model.module_list[40][1] = cspnet3.m[2].cv1.bn
    model.module_list[40][2] = cspnet3.m[2].cv1.act
    model.module_list[41][0] = cspnet3.m[2].cv2.conv
    model.module_list[41][1] = cspnet3.m[2].cv2.bn
    model.module_list[41][2] = cspnet3.m[2].cv2.act
    conv4 = list(modelyolov5.model.children())[7]
    model.module_list[47][0] = conv4.conv
    model.module_list[47][1] = conv4.bn
    model.module_list[47][2] = conv4.act
    spp = list(modelyolov5.model.children())[8]
    model.module_list[48][0] = spp.cv1.conv
    model.module_list[48][1] = spp.cv1.bn
    model.module_list[48][2] = spp.cv1.act
    model.module_list[49] = spp.m[0]
    model.module_list[51] = spp.m[1]
    model.module_list[53] = spp.m[2]
    model.module_list[55][0] = spp.cv2.conv
    model.module_list[55][1] = spp.cv2.bn
    model.module_list[55][2] = spp.cv2.act
    cspnet4 = list(modelyolov5.model.children())[9]
    model.module_list[56][0] = cspnet4.cv2
    model.module_list[58][0] = cspnet4.cv1.conv
    model.module_list[58][1] = cspnet4.cv1.bn
    model.module_list[58][2] = cspnet4.cv1.act
    model.module_list[61][0] = cspnet4.cv3
    model.module_list[63][0] = cspnet4.bn
    model.module_list[63][1] = cspnet4.act
    model.module_list[64][0] = cspnet4.cv4.conv
    model.module_list[64][1] = cspnet4.cv4.bn
    model.module_list[64][2] = cspnet4.cv4.act
    model.module_list[59][0] = cspnet4.m[0].cv1.conv
    model.module_list[59][1] = cspnet4.m[0].cv1.bn
    model.module_list[59][2] = cspnet4.m[0].cv1.act
    model.module_list[60][0] = cspnet4.m[0].cv2.conv
    model.module_list[60][1] = cspnet4.m[0].cv2.bn
    model.module_list[60][2] = cspnet4.m[0].cv2.act
    conv5 = list(modelyolov5.model.children())[10]
    model.module_list[65][0] = conv5.conv
    model.module_list[65][1] = conv5.bn
    model.module_list[65][2] = conv5.act
    upsample1 = list(modelyolov5.model.children())[11]
    model.module_list[66] = upsample1
    cspnet5 = list(modelyolov5.model.children())[13]
    model.module_list[68][0] = cspnet5.cv2
    model.module_list[70][0] = cspnet5.cv1.conv
    model.module_list[70][1] = cspnet5.cv1.bn
    model.module_list[70][2] = cspnet5.cv1.act
    model.module_list[73][0] = cspnet5.cv3
    model.module_list[75][0] = cspnet5.bn
    model.module_list[75][1] = cspnet5.act
    model.module_list[76][0] = cspnet5.cv4.conv
    model.module_list[76][1] = cspnet5.cv4.bn
    model.module_list[76][2] = cspnet5.cv4.act
    model.module_list[71][0] = cspnet5.m[0].cv1.conv
    model.module_list[71][1] = cspnet5.m[0].cv1.bn
    model.module_list[71][2] = cspnet5.m[0].cv1.act
    model.module_list[72][0] = cspnet5.m[0].cv2.conv
    model.module_list[72][1] = cspnet5.m[0].cv2.bn
    model.module_list[72][2] = cspnet5.m[0].cv2.act
    conv6 = list(modelyolov5.model.children())[14]
    model.module_list[77][0] = conv6.conv
    model.module_list[77][1] = conv6.bn
    model.module_list[77][2] = conv6.act
    upsample2 = list(modelyolov5.model.children())[15]
    model.module_list[78] = upsample2
    cspnet6 = list(modelyolov5.model.children())[17]
    model.module_list[80][0] = cspnet6.cv2
    model.module_list[82][0] = cspnet6.cv1.conv
    model.module_list[82][1] = cspnet6.cv1.bn
    model.module_list[82][2] = cspnet6.cv1.act
    model.module_list[85][0] = cspnet6.cv3
    model.module_list[87][0] = cspnet6.bn
    model.module_list[87][1] = cspnet6.act
    model.module_list[88][0] = cspnet6.cv4.conv
    model.module_list[88][1] = cspnet6.cv4.bn
    model.module_list[88][2] = cspnet6.cv4.act
    model.module_list[83][0] = cspnet6.m[0].cv1.conv
    model.module_list[83][1] = cspnet6.m[0].cv1.bn
    model.module_list[83][2] = cspnet6.m[0].cv1.act
    model.module_list[84][0] = cspnet6.m[0].cv2.conv
    model.module_list[84][1] = cspnet6.m[0].cv2.bn
    model.module_list[84][2] = cspnet6.m[0].cv2.act
    conv7 = list(modelyolov5.model.children())[18]
    model.module_list[92][0] = conv7.conv
    model.module_list[92][1] = conv7.bn
    model.module_list[92][2] = conv7.act
    cspnet7 = list(modelyolov5.model.children())[20]
    model.module_list[94][0] = cspnet7.cv2
    model.module_list[96][0] = cspnet7.cv1.conv
    model.module_list[96][1] = cspnet7.cv1.bn
    model.module_list[96][2] = cspnet7.cv1.act
    model.module_list[99][0] = cspnet7.cv3
    model.module_list[101][0] = cspnet7.bn
    model.module_list[101][1] = cspnet7.act
    model.module_list[102][0] = cspnet7.cv4.conv
    model.module_list[102][1] = cspnet7.cv4.bn
    model.module_list[102][2] = cspnet7.cv4.act
    model.module_list[97][0] = cspnet7.m[0].cv1.conv
    model.module_list[97][1] = cspnet7.m[0].cv1.bn
    model.module_list[97][2] = cspnet7.m[0].cv1.act
    model.module_list[98][0] = cspnet7.m[0].cv2.conv
    model.module_list[98][1] = cspnet7.m[0].cv2.bn
    model.module_list[98][2] = cspnet7.m[0].cv2.act
    conv8 = list(modelyolov5.model.children())[21]
    model.module_list[106][0] = conv8.conv
    model.module_list[106][1] = conv8.bn
    model.module_list[106][2] = conv8.act
    cspnet8 = list(modelyolov5.model.children())[23]
    model.module_list[108][0] = cspnet8.cv2
    model.module_list[110][0] = cspnet8.cv1.conv
    model.module_list[110][1] = cspnet8.cv1.bn
    model.module_list[110][2] = cspnet8.cv1.act
    model.module_list[113][0] = cspnet8.cv3
    model.module_list[115][0] = cspnet8.bn
    model.module_list[115][1] = cspnet8.act
    model.module_list[116][0] = cspnet8.cv4.conv
    model.module_list[116][1] = cspnet8.cv4.bn
    model.module_list[116][2] = cspnet8.cv4.act
    model.module_list[111][0] = cspnet8.m[0].cv1.conv
    model.module_list[111][1] = cspnet8.m[0].cv1.bn
    model.module_list[111][2] = cspnet8.m[0].cv1.act
    model.module_list[112][0] = cspnet8.m[0].cv2.conv
    model.module_list[112][1] = cspnet8.m[0].cv2.bn
    model.module_list[112][2] = cspnet8.m[0].cv2.act
    detect = list(modelyolov5.model.children())[24]
    model.module_list[89][0] = detect.m[0]
    model.module_list[103][0] = detect.m[1]
    model.module_list[117][0] = detect.m[2]

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov5s_v2.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/fangweisui.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov5s_v2.pt', help='sparse model weights')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)


    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #the way of loading yolov5s
    # ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    # modelyolov5 = Model('models/yolov5s_v3.yaml', nc=80).to(device)
    # exclude = ['anchor']  # exclude keys
    # ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
    #                  if k in modelyolov5.state_dict() and not any(x in k for x in exclude)
    #                  and modelyolov5.state_dict()[k].shape == v.shape}
    # modelyolov5.load_state_dict(ckpt['model'], strict=False)

    #another way of loading yolov5s
    modelyolov5=torch.load(opt.weights, map_location=device)['model'].float().eval()
    modelyolov5.model[24].export = False  # onnx export

    initialize_weights(modelyolov5)

   

    #load yolov5s from cfg
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)
    copy_weight(modelyolov5,model)

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



