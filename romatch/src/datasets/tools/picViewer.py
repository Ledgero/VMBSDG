import os
import sys
from tools import wandering_tool
from tools import fileProcess
from tools import pointProjector_debug
import numpy as np
import h5py
from tools import pose2RTK
import pprint
import time
import cv2

def depthViewer(depthImg, Thr, importBase, fileName):
    img_depth_vis = depthImg / Thr
    img_depth_vis[img_depth_vis > 1] = 1.
    # 3. 转换为整形
    img_depth_vis = (img_depth_vis * 255).astype(np.uint8)
    # 4. 保存为文件
    cv2.imwrite(os.path.join(importBase, fileName+'.png'), img_depth_vis)