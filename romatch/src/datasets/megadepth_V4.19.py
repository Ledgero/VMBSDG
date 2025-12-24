import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

import cv2
# import torchvision.transforms as transforms
from src.datasets.tools import wandering_tool
import random
import time

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth, read_megadepth_gray_forMixed, read_megadepth_depth_forMixed, resize_gray



class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        
        self.scene_info={}
        self.scene_infoBuf = np.load(npz_path, allow_pickle=True)
        self.scene_info['image_paths'] = self.scene_infoBuf['image_paths'].copy()
        self.scene_info['depth_paths'] = self.scene_infoBuf['depth_paths'].copy()
        self.scene_info['intrinsics'] = self.scene_infoBuf['intrinsics'].copy()
        self.scene_info['poses'] = self.scene_infoBuf['poses'].copy()
        self.pair_infos = self.scene_infoBuf['pair_infos'].copy()
        del self.scene_infoBuf #这里是新加的，不确定有没有问题
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1][0] > min_overlap_score]  

        #回头再加到从配置文件读取
        #，需要根据txt，读取所有npz list，给之后getItem调用，getItem里随机找图贴（可以把cpu线程高多一点）
        self.myADDnpz_path=npz_path.split('/')[-1]
        # print("MegaDepthDataset init ", self.myADDnpz_path)
        #mixedTxtPath="/root/data1/Loftr_base/LoFTR/data/megadepth/index/trainvaltest_list_redu/realReduced1To100.txt" #realReduced1To100Div3, realReduced1To100, realReduced2To100, realReduced1To100_200
        mixedTxtPath="/root/data1/Loftr_base/LoFTR/data/megadepth/index/trainvaltest_list/train_list.txt" 
        
        with open(mixedTxtPath,'r') as file:
            self.scene_infos_mixed = [line.rstrip('\n') for line in file]
        

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125) #这是个啥

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        #其实会调用多次，每次都会重新读取，每次直接重新贴图加载
       # (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]  #这里改一下数据组织方式把，不知道原来的megadepth是怎么办到的。
        [idx0F, idx1F] = self.pair_infos[idx][0]  #这里改一下数据组织方式把，不知道原来的megadepth是怎么办到的。
        (idx0, idx1)=[int(idx0F),int(idx1F)]
        #print("selfmodel: ",self.mode,"__getitem__ ", self.myADDnpz_path.split('/')[-1], "idx: ", idx)

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0, origWH0, image_noRsz0 = read_megadepth_gray_forMixed(
            img_name0, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1, origWH1, image_noRsz1 = read_megadepth_gray_forMixed(
            img_name1, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        # print("image_noRsz0 ",image_noRsz0.shape)
        # print("image_noRsz1 ",image_noRsz1.shape)
        # exit()

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0, origDepWH0, depth_noPad0 = read_megadepth_depth_forMixed(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
            depth1, origDepWH1, depth_noPad1 = read_megadepth_depth_forMixed(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size) #这里size和图像size不一样吗
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)、
        T_1to0 = T_0to1.inverse()

        mixed_Train=1 #这些代码需要整理一下
        if mixed_Train==1 and self.mode=='train': 

            random.seed(time.time()*1000) #设置随机数种子
            #random.seed(idx) #设置随机数种子
            #print("selfmodel: ",self.mode,"__getitem__ ", self.myADDnpz_path.split('/')[-1], "idx: ", idx)

            npz_BasePath_mixed="/root/data1/Loftr_base/LoFTR/data/megadepth/index/scene_info_0.1_0.7/" ##scene_info_0.1_0.7_reduced4 scene_info_0.1_0.7_reduced1 scene_info_0.1_0.7
            root_dir_mixed="/root/data1/Loftr_base/LoFTR/data/megadepth/train/" #换位置
            # sceneName= "0478_0.3_0.5" 
            sceneName = random.choice(self.scene_infos_mixed)

            self.scene_infoBuf_mixed = dict(np.load(npz_BasePath_mixed+sceneName+".npz", allow_pickle=True))
            scene_info_mixed=self.scene_infoBuf_mixed.copy()
            pair_infos_mixed = self.scene_infoBuf_mixed['pair_infos'].copy()
            del self.scene_infoBuf_mixed #释放空间

            idx_mixed=int(random.random()*(len(pair_infos_mixed)-1)) #改成随机的了，对于npz内多种数据有用
            [ridx0F, ridx1F] = pair_infos_mixed[idx_mixed][0] 
            (ridx0, ridx1) = [int(ridx0F),int(ridx1F)]
            # print("idx_mixed : ",idx_mixed,len(pair_infos_mixed), (ridx0, ridx1))

            # read grayscale image and mask. (1, h, w) and (h, w)
            rimg_name0 = osp.join(root_dir_mixed, scene_info_mixed['image_paths'][ridx0])
            rimg_name1 = osp.join(root_dir_mixed, scene_info_mixed['image_paths'][ridx1])
            
            # 这里改一下，直接输出原始图像，假设rimage0是原始图像, 注意rimage_noRsz0是numpy类型
            _, _, _ ,rOrigWH0, rimage_noRsz0 = read_megadepth_gray_forMixed(
                rimg_name0, self.img_resize, self.df, self.img_padding, None)
            _, _, _ ,rOrigWH1, rimage_noRsz1 = read_megadepth_gray_forMixed(
                rimg_name1, self.img_resize, self.df, self.img_padding, None)

            # 这里也改一下，直接读取原始depth，也不padding
            if self.mode in ['train', 'val']:
                _, rOrigDepWH0, rdepth_noPad0 = read_megadepth_depth_forMixed(
                    osp.join(root_dir_mixed, scene_info_mixed['depth_paths'][ridx0]), pad_to=self.depth_max_size)
                _, rOrigDepWH1, rdepth_noPad1 = read_megadepth_depth_forMixed(
                    osp.join(root_dir_mixed, scene_info_mixed['depth_paths'][ridx1]), pad_to=self.depth_max_size) #这里size和图像size不一样吗
            else:
                rdepth0 = rdepth1 = torch.tensor([])

            # read intrinsics of original size
            rK_0 = torch.tensor(scene_info_mixed['intrinsics'][ridx0].copy(), dtype=torch.float).reshape(3, 3)
            rK_1 = torch.tensor(scene_info_mixed['intrinsics'][ridx1].copy(), dtype=torch.float).reshape(3, 3)

            # read and compute relative poses
            rT0 = scene_info_mixed['poses'][ridx0]
            rT1 = scene_info_mixed['poses'][ridx1]
            rT_0to1 = torch.tensor(np.matmul(rT1, np.linalg.inv(rT0)), dtype=torch.float)[:4, :4]  # (4, 4)、
            rT_1to0 = rT_0to1.inverse()


            # #只是为了完成对应点匹配的match效果，这里只针对合成数据分辨率origWH0=origWH1
            # _, MapX0, MapY0 = wandering_tool.findMap(K_0, K_1, T_0to1, depth_noPad0, depth_noPad1, origWH0.numpy().astype(int))
            # Img1X ,PosHW0=wandering_tool.nearest_non_zero(MapX0) #注意是数组坐标H W,先行id 再列id
            # Img1Y = MapY0[PosHW0[0],PosHW0[1]]
            # #WH顺序
            # conrPtsTL=np.array([[PosHW0[1],PosHW0[0]], [int(Img1X),int(Img1Y)]]) #左上角最小坐标, 之后试一下这个全图随机。或者靠近最上面随机
            # #加一个限制，conrPtsTL点不能太靠下靠右,不能超过2/3区域
            # if conrPtsTL[0][0]/origWH0[0]>2/3 or conrPtsTL[0][1]/origWH0[1]>2/3 or conrPtsTL[1][0]/origWH1[0]>2/3 or conrPtsTL[1][1]/origWH1[1]>2/3:
            #     conrPtsTL=np.array([[0,0], [0,0]])
            # Base0=random.uniform(0.02,1)
            # Base1=random.uniform(0.02,1)
            # reScaleBase = [Base0, Base1] #下一步优化
            # scaleWH0=[origWH0[0]-conrPtsTL[0][0],origWH0[1]-conrPtsTL[0][1]] #缩放比较的尺寸，要不是整张大图，要不是匹配点到右下角的图
            # scaleWH1=[origWH1[0]-conrPtsTL[1][0],origWH1[1]-conrPtsTL[1][1]]
            # rScale0_mixed = max(rOrigWH0[0]/(reScaleBase[0]*scaleWH0[0]), rOrigWH0[1]/(reScaleBase[0]*scaleWH0[1])) #真实图WH/(缩小到合成倍率*合成WH)=真实大图/真实小图
            # rScale1_mixed = max(rOrigWH1[0]/(reScaleBase[1]*scaleWH1[0]), rOrigWH1[1]/(reScaleBase[1]*scaleWH1[1]))


             #原始 对real图像做缩小，按照最小边缩放
            conrPtsTL=np.array([[0,0], [0,0]]) #左上角最小坐标 这个img0 img1不同 后期优化一下，动这个其实也得调试，K的搭配不一定对
            Base0=random.uniform(0.02,1)
            Base1=random.uniform(0.02,1)
            # Base0=0.3
            # Base1=0.3
            reScaleBase = [Base0, Base1]
            rScale0_mixed = max(rOrigWH0[0]/(reScaleBase[0]*origWH0[0]), rOrigWH0[1]/(reScaleBase[0]*origWH0[1])) #真实图WH/(缩小到合成倍率*合成WH)=真实大图/真实小图
            rScale1_mixed = max(rOrigWH1[0]/(reScaleBase[1]*origWH1[0]), rOrigWH1[1]/(reScaleBase[1]*origWH1[1]))

            #图像缩放
            M0=np.float32([[1/rScale0_mixed, 0, 0],[0, 1/rScale0_mixed, 0]]) #这里conrPtsTL要不要乘系数
            rimage0_trans=cv2.warpAffine(rimage_noRsz0,M0,(int(rOrigWH0[0]/rScale0_mixed),int(rOrigWH0[1]/rScale0_mixed)))
            rdepth0_trans=cv2.warpAffine(rdepth_noPad0,M0,(int(rOrigDepWH0[0]/rScale0_mixed),int(rOrigDepWH0[1]/rScale0_mixed)))
            M1=np.float32([[1/rScale1_mixed, 0, 0],[0, 1/rScale1_mixed, 0]])
            rimage1_trans=cv2.warpAffine(rimage_noRsz1,M1,(int(rOrigWH1[0]/rScale1_mixed),int(rOrigWH1[1]/rScale1_mixed)))
            rdepth1_trans=cv2.warpAffine(rdepth_noPad1,M1,(int(rOrigDepWH1[0]/rScale1_mixed),int(rOrigDepWH1[1]/rScale1_mixed)))
            #深度图resize缩小，不同的插值方式会不会影响深度计算

            #图像嵌入 #记得把图像转tensor,且加一个维度 [None] 做这样的处理 .from_numpy(image).float()[None] / 255 
            img0_Mixed=image_noRsz0.copy()
            img1_Mixed=image_noRsz1.copy()

            img0_Mixed[conrPtsTL[0][1]: conrPtsTL[0][1]+rimage0_trans.shape[0], conrPtsTL[0][0]: conrPtsTL[0][0]+rimage0_trans.shape[1]]=rimage0_trans
            img1_Mixed[conrPtsTL[1][1]: conrPtsTL[1][1]+rimage1_trans.shape[0], conrPtsTL[1][0]: conrPtsTL[1][0]+rimage1_trans.shape[1]]=rimage1_trans
            depth0_Mixed=depth0
            depth1_Mixed=depth1
            depth0_Mixed[conrPtsTL[0][1]: conrPtsTL[0][1]+rdepth0_trans.shape[0], conrPtsTL[0][0]: conrPtsTL[0][0]+rdepth0_trans.shape[1]]=torch.from_numpy(rdepth0_trans) #这一步已经pad了
            depth1_Mixed[conrPtsTL[1][1]: conrPtsTL[1][1]+rdepth1_trans.shape[0], conrPtsTL[1][0]: conrPtsTL[1][0]+rdepth1_trans.shape[1]]=torch.from_numpy(rdepth1_trans) #这一步已经pad了
            # cv2.imwrite("/home/smf/Airsim/LoFTR/logs/tb_logs/debug_buf/"+str(idx)+"img0_Mixed.jpg",img0_Mixed.astype(np.uint8))
            # cv2.imwrite("/home/smf/Airsim/LoFTR/logs/tb_logs/debug_buf/"+str(idx)+"img1_Mixed.jpg",img1_Mixed.astype(np.uint8))

            #K值更新，这里存疑，conrPtsTL不知道是不是那样
            rK0=torch.clone(rK_0)
            rK1=torch.clone(rK_1)
            rK0[0:2,0:3]=rK0[0:2,0:3]*(1/rScale0_mixed)
            rK0[0][2], rK0[1][2]=rK0[0][2]+conrPtsTL[0][0], rK0[1][2]+conrPtsTL[0][1]
            rK1[0:2,0:3]=rK1[0:2,0:3]*(1/rScale1_mixed)
            rK1[0][2], rK1[1][2]=rK1[0][2]+conrPtsTL[1][0], rK1[1][2]+conrPtsTL[1][1]
            

            # #test 这个一会test 一下，可能要封装picCheck整体，以支持对不同图片的pad和check
            #匹配似乎有点偏差，特别是加了平移的时候，要看看原图是不是这样
            # imgTest=wandering_tool.getCheckImgComb(img0_Mixed,img1_Mixed,depth0_Mixed,depth1_Mixed,rK0, rK1, rT_0to1, rT_1to0)
            # imgTest=wandering_tool.getCheckImgComb(img0_Mixed,img1_Mixed,depth0_Mixed,depth1_Mixed,K_0, K_1, T_0to1, T_1to0)
            # cv2.imwrite("/home/smf/Airsim/LoFTR/imgTest1.jpg",imgTest)

            #resize, img按照之前的 resize 转tensor,这里直接用image0 可能有风险
            image0, mask0, scale0=resize_gray(img0_Mixed, self.img_resize, self.df, self.img_padding, None)
            image1, mask1, scale1=resize_gray(img1_Mixed, self.img_resize, self.df, self.img_padding, None)
            conrPtsTL=torch.from_numpy(conrPtsTL)
            r0WH=[int(rOrigWH0[0]/rScale0_mixed),int(rOrigWH0[1]/rScale0_mixed)]
            r1WH=[int(rOrigWH1[0]/rScale1_mixed),int(rOrigWH1[1]/rScale1_mixed)]
            rWH=torch.tensor([r0WH,r1WH]) #这个是没scale过的
            # print("rWH ",rWH)

            data = {
            'image0': image0,  # (1, h, w) 加cuda
            'depth0': depth0_Mixed,  # (h, w)
            'image1': image1,
            'depth1': depth1_Mixed,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            
            #adding for mix 如果这么加超内存的话就再切换
            'rT_0to1': rT_0to1,  # (4, 4)
            'rT_1to0': rT_1to0,
            'rK0': rK0,  # (3, 3)
            'rK1': rK1,
            'conrPtsTL': conrPtsTL,
            'rWH':rWH,
            }

        else:#原始data更新
            data = {
            'image0': image0,  # (1, h, w) 加cuda
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data

    