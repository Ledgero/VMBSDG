import torch


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()
    
    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2 #这里应该是深度阈值，得要比较一下。有可能这里很少
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0


@torch.no_grad()
def warp_kpts_withSubWin(kpts0, depth0, depth1, T_0to1, K0, K1,rT_0to1, rK0, rK1, img0P0, img0P1, img1P0, img1P1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt 这里解释要改一下
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    #这里计算稍复杂，后面加注释

    def inSubWin(pt, PTL, PBR): #在窗口内的有效区间[TL,BR)
        resbuf=(pt >= PTL) & (pt < PBR)
        return resbuf[...,0] & resbuf[...,1]

    kpts0_inSubWin0=inSubWin(kpts0,img0P0,img0P1)

    kptsOrig=torch.clone(kpts0)
    kptsOrig[kpts0_inSubWin0]=0
    kptsSubWin=torch.clone(kpts0)
    kptsSubWin[~kpts0_inSubWin0]=0
    #对kptsOrig kptsSubWin分别做计算 然后融合到一块去，调用这个来算
    _ , w_pt0_i_orig = warp_kpts(kptsOrig, depth0, depth1, T_0to1, K0, K1)
    _ , w_pt0_i_subWin = warp_kpts(kptsSubWin, depth0, depth1, rT_0to1, rK0, rK1)
    #检查变换完之后在不在小图上，w_pt0_i_subWin在img1小图里，w_pt0_i_orig不在img1小图里,落到不该落得区间就置0,0
    w_pt0_i_orig_inSubWin1=inSubWin(w_pt0_i_orig,img1P0,img1P1)
    w_pt0_i_orig[w_pt0_i_orig_inSubWin1]=0
    w_pt0_i_subWin_inSubWin1=inSubWin(w_pt0_i_subWin,img1P0,img1P1)
    w_pt0_i_subWin[~w_pt0_i_subWin_inSubWin1]=0
    #小图对应点在小图上，大图对应点在大图上
    w_pt0=torch.where(torch.unsqueeze(kpts0_inSubWin0,2).repeat(1,1,2),w_pt0_i_subWin,w_pt0_i_orig)
    
    #这里先屏蔽掉大图的匹配
    # w_pt0_i_subWin[~pts_inSubWin]=0
    # w_pt0=w_pt0_i_subWin

    #深度check 不知道是不是在经过mix之后依旧有效，可能要检查一下
    return _ , w_pt0

#无多batch适配
@torch.no_grad()
def warp_kpts_withSubWin_singleBtc(kpts0, depth0, depth1, T_0to1, K0, K1,rT_0to1, rK0, rK1, conrPtsTL0, rWH0, conrPtsTL1, rWH1): #img0 img1已经混合好了, depth0 1也混合好了
    """ Warp kpts0 from I0 to I1 with depth, K and Rt 这里解释要改一下
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    #无多batch适配
    def inSubWin(pt, TLx, TLy, BRx, BRy): #在窗口内的有效区间[TL,BR)
        return (pt[..., 0] >= TLx) & (pt[..., 0] < BRx) & (pt[..., 1] >= TLy) & (pt[..., 1] < BRy)
    kpts0_inSubWin0=inSubWin(kpts0,conrPtsTL0[0],conrPtsTL0[1],conrPtsTL0[0]+rWH0[0],conrPtsTL0[1]+rWH0[1])

    kptsOrig=torch.clone(kpts0)
    kptsOrig[kpts0_inSubWin0]=0
    kptsSubWin=torch.clone(kpts0)
    kptsSubWin[~kpts0_inSubWin0]=0
    #对kptsOrig kptsSubWin分别做计算 然后融合到一块去，调用这个来算
    _ , w_pt0_i_orig = warp_kpts(kptsOrig, depth0, depth1, T_0to1, K0, K1)
    _ , w_pt0_i_subWin = warp_kpts(kptsSubWin, depth0, depth1, rT_0to1, rK0, rK1)
    #检查变换完之后在不在小图上，w_pt0_i_subWin在img1小图里，w_pt0_i_orig不在img1小图里,落到不该落得区间就置0,0
    w_pt0_i_orig_inSubWin1=inSubWin(w_pt0_i_orig,conrPtsTL1[0],conrPtsTL1[1],conrPtsTL1[0]+rWH1[0],conrPtsTL1[1]+rWH1[1])
    w_pt0_i_orig[w_pt0_i_orig_inSubWin1]=0
    w_pt0_i_subWin_inSubWin1=inSubWin(w_pt0_i_subWin,conrPtsTL1[0],conrPtsTL1[1],conrPtsTL1[0]+rWH1[0],conrPtsTL1[1]+rWH1[1])
    w_pt0_i_subWin[~w_pt0_i_subWin_inSubWin1]=0
    #小图对应点在小图上，大图对应点在大图上
    w_pt0=torch.where(torch.unsqueeze(kpts0_inSubWin0,2).repeat(1,1,2),w_pt0_i_subWin,w_pt0_i_orig)
    
    #这里先屏蔽掉大图的匹配
    # w_pt0_i_subWin[~pts_inSubWin]=0
    # w_pt0=w_pt0_i_subWin

    #深度check 不知道是不是在经过mix之后依旧有效，可能要检查一下
    return _ , w_pt0