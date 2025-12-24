import math

# import airsim
import time

import numpy
import numpy as np
import scipy.sparse as sparse
# import igl
# from meshplot import plot, subplot, interact
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path

import cv2
import time
import sys
import random
import glob
# from airsim import *
import h5py

np.set_printoptions(suppress=True)

def project_screen_to_3d_point(UV, K, R, T, Depth):
    UV = np.concatenate([UV, [[1]]])
    RTrans=np.transpose(R)
    KTrans=np.linalg.inv(K)
    Pw=np.matmul(RTrans,Depth*np.matmul(KTrans,UV)-T);
    return Pw

def project_3d_point_to_screen(Pw, K, R, T):

    CamPoint = np.matmul(R, Pw) + T
    Depth = CamPoint[2]

    ScreenPoint = np.matmul(K, CamPoint)
    ScreenPoint = ScreenPoint / Depth

    if Depth == 0:
        return None

    return np.array([
        ScreenPoint[0],
        ScreenPoint[1],
        Depth
    ]).reshape(3, )

def convertI1toI2(K1,R1,T1,UV1,Depth1,K2,R2,T2,UV2,Depth2): #把I1投影到I2上
    Pw=project_screen_to_3d_point(UV1,K1,R1,T1,Depth1)
    UV2_pro=project_3d_point_to_screen(Pw,K2,R2,T2)
    print("compute UV2 res: ",  UV2_pro)
    print("GT UV2 is ",str(UV2)+" "+str(Depth2))

def check():
    folder_path = "D:/DataSet/Loftr数据集/index/megadepth_indices/scene_info_val_1500/0022_0.1_0.3.npz"
    # folder_path = "D:\\Users\\forev\\PycharmProjects\\AirSim_Mountain\\DataSetProcess\\data\\pic\\index\\scene0.npz"
    datas = np.load(folder_path, allow_pickle=True)
    for key in datas.keys():
        print(key)

    idx1f=datas['pair_infos'][0][0][0]
    idx2f=datas['pair_infos'][0][0][1]
    idx1=int(idx1f)
    idx2 = int(idx2f)

    image_path1=datas['image_paths'][idx1]
    depth_path1=datas['depth_paths'][idx1]
    intrinsic1=datas['intrinsics'][idx1]
    pose1=datas['poses'][idx1]

    image_path2 = datas['image_paths'][idx2]
    depth_path2 = datas['depth_paths'][idx2]
    intrinsic2 = datas['intrinsics'][idx2]
    pose2 = datas['poses'][idx2]

    K1=intrinsic1
    K2=intrinsic2

    R1=pose1[:3,:3]
    T1=pose1[:3,3:]
    R2 = pose2[:3, :3]
    T2 = pose2[:3, 3:]
    # print("K1",K1)
    # print("R1",R1)
    # print("T1",T1)

    print("img: ",image_path1+" "+image_path2)
    print("depth: ",depth_path1+" "+depth_path2)

    UV1=np.array([[625],[470]])
    UV2 = np.array([[739], [549]])

    # h5_file1 = "D:\\Users\\forev\\PycharmProjects\\AirSim_Mountain\\DataSetProcess\\data\\pic\\0\\depth\\img0.h5"
    h5_file1 = "D:\\Users\\forev\\PycharmProjects\\AirSim_Mountain\\DataSetProcess\\buf\\427154679_de14c315f4_o.h5"
    h5file1 = h5py.File(h5_file1, "r")
    depths1=h5file1['depth'][:]
    depth1=depths1[UV1[1][0]][UV1[0][0]]

    # h5_file2 = "D:\\Users\\forev\\PycharmProjects\\AirSim_Mountain\\DataSetProcess\\data\\pic\\0\\depth\\img1.h5"
    h5_file2 = "D:\\Users\\forev\\PycharmProjects\\AirSim_Mountain\\DataSetProcess\\buf\\3553841868_b6ee93bf43_o.h5"
    h5file2 = h5py.File(h5_file2, "r")
    depths2 = h5file2['depth'][:]
    depth2 = depths2[UV2[1][0]][UV2[0][0]]

    # Pw=project_screen_to_3d_point(UV1,K1,R1,T1,depth1)
    # PUV=project_3d_point_to_screen(Pw,K1,R1,T1)
    # print("Pw: ",Pw)
    # print("PUV",PUV)

    convertI1toI2(K1,R1,T1,UV1,depth1,K2,R2,T2,UV2,depth2);
    #convertI1toI2(K2, R2, T2, UV2, depth2,K1, R1, T1, UV1, depth1);


def check_CityDataSet(K1,K2,R1,R2,T1,T2,depth1,depth2,UV1,UV2):#检查I1投影到I2，是否与传入UV2一致

    Pw=project_screen_to_3d_point(UV1,K1,R1,T1,depth1)
    print("Pw: ",Pw)

    convertI1toI2(K1,R1,T1,UV1,depth1,K2,R2,T2,UV2,depth2);
    #convertI1toI2(K2, R2, T2, UV2, depth2,K1, R1, T1, UV1, depth1);


def check_CityDataSetDebug(K1, K2 ,pose1, pose2, depths1, depths2, UV1, UV2):  # 检查I1投影到I2，是否与传入UV2一致

    R1=pose1[:3,:3]
    T1=pose1[:3,3:]
    R2 = pose2[:3, :3]
    T2 = pose2[:3, 3:]
    depth1 = depths1[UV1[1][0]][UV1[0][0]]
    depth2 = depths2[UV2[1][0]][UV2[0][0]]
    # print("debug depth0: ", depth1)
    # np.savetxt('debugTxt.txt',depths1[303:322,471:513],fmt='%.3f')

    Pw = project_screen_to_3d_point(UV1, K1, R1, T1, depth1)
    print("Pw: ", Pw)

    convertI1toI2(K1, R1, T1, UV1, depth1, K2, R2, T2, UV2, depth2);


def check_CityDataSet(K1, K2 ,pose1, pose2, depths1, depths2, UV1, UV2):  # 检查I1投影到I2，是否与传入UV2一致

    R1=pose1[:3,:3]
    T1=pose1[:3,3:]
    R2 = pose2[:3, :3]
    T2 = pose2[:3, 3:]
    depth1 = depths1[UV1[1][0]][UV1[0][0]]
    depth2 = depths2[UV2[1][0]][UV2[0][0]]

    Pw = project_screen_to_3d_point(UV1, K1, R1, T1, depth1)
    print("Pw: ", Pw)

    convertI1toI2(K1, R1, T1, UV1, depth1, K2, R2, T2, UV2, depth2);

