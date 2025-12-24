from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid

from src.utils.plotting import make_matching_figure

#调试绘图
from .geometry import warp_kpts, warp_kpts_withSubWin,warp_kpts_withSubWin_singleBtc
import numpy as np
import matplotlib.pyplot as plt
import cv2

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape #修改的话从这里开始处理,这里是1会不会变大？
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0] #又缩小了8倍 从（640，640）-->(80, 80)
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    # if N>=2: #我加的，
    #     print("注意 N 等于2 N ",N)
    #     exit(0)
    # print("N： ", N," data['image0'] ",data['image0'].shape)

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0']) #掩盖掉，可能是赋值0
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    if data.get('rT_0to1') is None: #正常train val阶段，mix val阶段
        _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
        _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    else: #mix train
        # # 单batch处理
        # _ , w_pt0_i0 = warp_kpts_withSubWin_singleBtc(grid_pt0_i,  data['depth0'], data['depth1'], \
        #     data['T_0to1'], data['K0'], data['K1'],data['rT_0to1'], data['rK0'], data['rK1'], \
        #     data['conrPtsTL'][0][0], data['rWH'][0][0],data['conrPtsTL'][0][1], data['rWH'][0][1])
        # _ , w_pt1_i0 = warp_kpts_withSubWin_singleBtc(grid_pt1_i,  data['depth1'], data['depth0'], \
        #     data['T_1to0'], data['K1'], data['K0'],data['rT_1to0'], data['rK1'], data['rK0'], \
        #     data['conrPtsTL'][0][1], data['rWH'][0][1],data['conrPtsTL'][0][0], data['rWH'][0][0]) 

        conrPtsBR=data['conrPtsTL']+data['rWH']
        img0P0, img1P0=data['conrPtsTL'][:,0:1,:], data['conrPtsTL'][:,1:2,:]#P0左上角的0
        img0P1, img1P1=conrPtsBR[:,0:1,:], conrPtsBR[:,1:2,:] #P1右下角的点
        _ , w_pt0_i = warp_kpts_withSubWin(grid_pt0_i,  data['depth0'], data['depth1'], \
            data['T_0to1'], data['K0'], data['K1'],data['rT_0to1'], data['rK0'], data['rK1'], \
            img0P0, img0P1, img1P0, img1P1)
        _ , w_pt1_i = warp_kpts_withSubWin(grid_pt1_i,  data['depth1'], data['depth0'], \
            data['T_1to0'], data['K1'], data['K0'],data['rT_1to0'], data['rK1'], data['rK0'], \
            img1P0, img1P1, img0P0, img0P1)     
        
        # # 多batch 调试，测试是不是和单batch一样
        # print(torch.equal(w_pt0_i0,w_pt0_i),torch.equal(w_pt1_i0,w_pt1_i))
        # mask = w_pt0_i != w_pt0_i0
        # # 打印出不同之处
        # indices = torch.nonzero(mask)
        # print(indices)
        # print("data['rWH'] ",data['rWH'])
        # exit(0)

    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1 
    #数组id是img0像素id，值是对应img1地像素id
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    #需要小心我的合成数据相互检测，是不是检测相互点很少？要看看有多少

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0 #[1,L]
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0 

    #loop back存储两两匹配上 以img0 id为索引，找img1 对应id, 再找回img0 id。
    #如果两个img0 id相等，那就是img0对应id点双向匹配
    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1) #id为img0 对应值为该点是否匹配上
    correct_0to1[:, 0] = False  # ignore the top-left corner,把左上角0 0置无效

    # #暂时先不弄，先测试移动到左上角
    #如果要根据匹配点来的话，可能要在这里改
    
    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids] 

    # if b_ids[0]!=0:#b_ids没做适配会不会有问题 目前看还好。是不是多机代码，加一个预防
    #     print("bad: ",b_ids,"bids eorrors-----")
    #     exit()
    
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # #画调试图、这里感觉不太对，原始的都对不上？，所以不一定是我的有问题。
    # #这个之后再test把，
    # print("data['image0'] ",data['image0'].shape)
    # print("data['image1'] ",data['image1'].shape)
    # b_id=0
    # img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8) #把图像缩小
    # img0 = cv2.resize(img0,(w0,h0))
    # img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    # img1 = cv2.resize(img1,(w1,h1))
    # # kpts0 = np.stack((i_ids.cpu().numpy() % w0, i_ids.cpu().numpy() // w0),axis=1)
    # # kpts1 = np.stack((j_ids.cpu().numpy() % w1, i_ids.cpu().numpy() // w1),axis=1)
    # color = np.ones((grid_pt0_i.shape[1], 4)) * np.array([1,0,0,0.3])
    # figure = make_matching_figure(img0, img1, grid_pt0_i.cpu().numpy()[0], w_pt0_i.cpu().numpy()[0], color)
    # figure.savefig("/home/smf/Airsim/LoFTR/matching_figure_myComputeNew.jpg")
    # exit()

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc 确实只是用尺度之前的w_pt0_i就能用来细粒度监督
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['LOFTR']['RESOLUTION'][1]
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError
