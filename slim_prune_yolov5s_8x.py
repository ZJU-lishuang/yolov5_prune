from modelsori import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse
import torchvision

from utils.model_transfer import copy_weight_v6,copy_weight_v6x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov5s_v6_hand.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/oxfordhand.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last_v6s.pt', help='sparse model weights')
    parser.add_argument('--global_percent', type=float, default=0.6, help='global channel prune percent')
    parser.add_argument('--layer_keep', type=float, default=0.01, help='channel keep percent per layer')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    modelyolov5 = torch.load(opt.weights, map_location=device)['model'].float()  # load FP32 model
    stride=32.0
    if len(modelyolov5.yaml["anchors"]) == 4:
        copy_weight_v6x(modelyolov5, model)
        stride=64.0
    else:
        copy_weight_v6(modelyolov5, model)

    eval_model = lambda model:test(model=model,cfg=opt.cfg, data=opt.data, batch_size=4, img_size=img_size,stride=stride)
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    print("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)

    CBL_idx, Conv_idx, prune_idx, _, _= parse_module_defs2(model.module_defs)



    bn_weights = gather_bn_weights(model.module_list, prune_idx)

    sorted_bn = torch.sort(bn_weights)[0]
    sorted_bn, sorted_index = torch.sort(bn_weights)
    thresh_index = int(len(bn_weights) * opt.global_percent)
    thresh = sorted_bn[thresh_index].cuda()

    print(f'Global Threshold should be less than {thresh:.4f}.')

    #%%
    def obtain_filters_mask(model, thre, CBL_idx, prune_idx):

        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        for idx in CBL_idx:
            # bn_module = model.module_list[idx][1]
            bn_module = model.module_list[idx][1] if type(
                model.module_list[idx][1]).__name__ == 'BatchNorm2d' else model.module_list[idx][0]
            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                if model.module_defs[idx][ 'type'] == 'convolutional_noconv':
                    channels = weight_copy.shape[0]
                    channels_half=int(channels/2)
                    weight_copy1=weight_copy[:channels_half]
                    weight_copy2 = weight_copy[channels_half:]
                    min_channel_num = int(channels_half * opt.layer_keep) if int(channels_half * opt.layer_keep) > 0 else 1
                    mask1 = weight_copy1.gt(thresh).float()
                    mask2 = weight_copy2.gt(thresh).float()

                    if int(torch.sum(mask1)) < min_channel_num:
                        _, sorted_index_weights1 = torch.sort(weight_copy1, descending=True)
                        mask1[sorted_index_weights1[:min_channel_num]] = 1.

                    if int(torch.sum(mask2)) < min_channel_num:
                        _, sorted_index_weights2 = torch.sort(weight_copy2, descending=True)
                        mask2[sorted_index_weights2[:min_channel_num]] = 1.

                    # regular
                    mask_cnt1 = int(mask1.sum())
                    mask_cnt2 = int(mask2.sum())

                    if mask_cnt1 % 8 != 0:
                        mask_cnt1 = int((mask_cnt1 // 8 + 1) * 8)
                    if mask_cnt2 % 8 != 0:
                        mask_cnt2 = int((mask_cnt2 // 8 + 1) * 8)

                    this_layer_sort_bn = bn_module.weight.data.abs().clone()
                    this_layer_sort_bn1 = this_layer_sort_bn[:channels_half]
                    this_layer_sort_bn2 = this_layer_sort_bn[channels_half:]
                    _, sorted_index_weights1 = torch.sort(this_layer_sort_bn1, descending=True)
                    _, sorted_index_weights2 = torch.sort(this_layer_sort_bn2, descending=True)
                    mask1[sorted_index_weights1[:mask_cnt1]] = 1.
                    mask2[sorted_index_weights2[:mask_cnt2]] = 1.


                    remain1 = int(mask1.sum())
                    pruned = pruned + mask1.shape[0] - remain1
                    remain2 = int(mask2.sum())
                    pruned = pruned + mask2.shape[0] - remain2

                    mask=torch.cat((mask1,mask2))
                    remain=remain1+remain2

                    print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                          f'remaining channel: {remain:>4d}')
                else:
                
                    channels = weight_copy.shape[0] #
                    min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 1
                    mask = weight_copy.gt(thresh).float()

                    if int(torch.sum(mask)) < min_channel_num:
                        _, sorted_index_weights = torch.sort(weight_copy,descending=True)
                        mask[sorted_index_weights[:min_channel_num]]=1.

                    # regular
                    mask_cnt = int(mask.sum())

                    if mask_cnt % 8 !=0:
                        mask_cnt=int((mask_cnt//8+1)*8)

                    this_layer_sort_bn = bn_module.weight.data.abs().clone()
                    _, sorted_index_weights = torch.sort(this_layer_sort_bn,descending=True)
                    mask[sorted_index_weights[:mask_cnt]]=1.

                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain

                    print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                            f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.clone())

        prune_ratio = pruned / total
        print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

        return num_filters, filters_mask

    num_filters, filters_mask = obtain_filters_mask(model, thresh, CBL_idx, prune_idx)
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i['is_access'] = False

    print('merge the mask of layers connected to shortcut!')
    merge_mask_regular(model, CBLidx2mask, CBLidx2filters)

    def prune_and_eval(model, CBL_idx, CBLidx2mask):
        model_copy = deepcopy(model)

        for idx in CBL_idx:
            # bn_module = model_copy.module_list[idx][1]
            bn_module = model_copy.module_list[idx][1] if type(
                model_copy.module_list[idx][1]).__name__ == 'BatchNorm2d' else model_copy.module_list[idx][0]
            mask = CBLidx2mask[idx].cuda()
            bn_module.weight.data.mul_(mask)

        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]

        print(f'mask the gamma as zero, mAP of the model is {mAP:.4f}')


    prune_and_eval(model, CBL_idx, CBLidx2mask)


    for i in CBLidx2mask:
        CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()

    pruned_model = prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask)
    print("\nnow prune the model but keep size,(actually add offset of BN beta to following layers), let's see how the mAP goes")

    with torch.no_grad():
        eval_model(pruned_model)

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i.pop('is_access')

    compact_module_defs = deepcopy(model.module_defs)
    for idx in CBL_idx:
        assert compact_module_defs[idx]['type'] == 'convolutional' or compact_module_defs[idx][
            'type'] == 'convolutional_noconv'
        num=CBLidx2filters[idx]
        compact_module_defs[idx]['filters'] = str(num)
        if compact_module_defs[idx]['type'] == 'convolutional_noconv':
            model_def = compact_module_defs[idx - 1]  # route
            assert compact_module_defs[idx - 1]['type'] == 'route'
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            assert compact_module_defs[idx - 1 + from_layers[0]]['type'] == 'convolutional_nobias'
            assert compact_module_defs[idx - 1 + from_layers[1] if from_layers[1] < 0 else from_layers[1]][
                       'type'] == 'convolutional_nobias'
            half_num = int(len(CBLidx2mask[idx]) / 2)
            mask1 = CBLidx2mask[idx][:half_num]
            mask2 = CBLidx2mask[idx][half_num:]
            remain1 = int(mask1.sum())
            remain2 = int(mask2.sum())
            compact_module_defs[idx - 1 + from_layers[0]]['filters'] = remain1
            compact_module_defs[idx - 1 + from_layers[1] if from_layers[1] < 0 else from_layers[1]]['filters'] = remain2


    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)


    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):
        # model.to('cpu').fuse()
        # model.module_list.to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)[0]
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output

    print('testing inference time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    diff = (pruned_output - compact_output).abs().gt(0.001).sum().item()
    if diff > 0:
        print('Something wrong with the pruned model!')

    print('testing the final model...')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)


    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)



    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{opt.global_percent}_keep_{opt.layer_keep}_8x_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    compact_model_name = opt.weights.replace('/', f'/prune_{opt.global_percent}_keep_{opt.layer_keep}_8x_')
    if compact_model_name.endswith('.pt'):
        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': compact_model.state_dict(),
                 # 'model': compact_model.module_list,  #部署调试加载的模型
                 'optimizer': None}
        torch.save(chkpt, compact_model_name)
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    # save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')

