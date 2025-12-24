import math

import airsim
import time

import numpy
import numpy as np
import scipy.sparse as sparse
import igl
from meshplot import plot, subplot, interact
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path

import cv2
import time
import sys
import random
import glob
from airsim import *

np.set_printoptions(suppress=True)

def rotation_matrix_from_angles(pry):
    pitch = pry[0]
    roll = pry[1]
    yaw = pry[2]
    sy = np.sin(yaw)
    cy = np.cos(yaw)
    sp = np.sin(pitch)
    cp = np.cos(pitch)
    sr = np.sin(roll)
    cr = np.cos(roll)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    # Roll is applied first, then pitch, then yaw.
    RyRx = np.matmul(Ry, Rx)
    return np.matmul(Rz, RyRx)


def project_3d_point_to_screen(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageWidthHeight):
    print("subjectXYZ point:",subjectXYZ)
    print("camXYZ point:", camXYZ)
    # Turn the camera position into a column vector.
    camPosition = np.transpose([camXYZ])

    # Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    pitchRollYaw = utils.to_eularian_angles(camQuaternion)

    # Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = rotation_matrix_from_angles(pitchRollYaw)

    # Change coordinates to get subjectXYZ in the camera's local coordinate system.
    XYZW = np.transpose([subjectXYZ])
    #XYZW = np.transpose([[27.14-108.33, 43.45+37.01, 0.8+36.73]])
    XYZW = np.add(XYZW, -camPosition) #这里表示是，先偏移，再翻转R（）
    XYZW = np.matmul(np.transpose(camRotation), XYZW)

    # Recreate the perspective projection of the camera.
    XYZW = np.concatenate([XYZW, [[1]]])
    XYZW = np.matmul(camProjMatrix4x4, XYZW)
    Depth=XYZW[3]
    if Depth==0:
        return None
    XYZW = XYZW / XYZW[3]

    # Move origin to the upper-left corner of the screen and multiply by size to get pixel values. Note that screen is in y,-z plane.
    normX = (1 - XYZW[0]) / 2
    normY = (1 + XYZW[1]) / 2
    # print(np.array([
    #     imageWidthHeight[0] * normX,
    #     imageWidthHeight[1] * normY,
    #     Depth
    # ]).reshape(3, ))

    return np.array([
        imageWidthHeight[0] * normX,
        imageWidthHeight[1] * normY,
        Depth
    ]).reshape(3, )


def project_3d_point_to_screen_By_RTK(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageWidthHeight,fov):

    fx=imageWidthHeight[0]/(2*math.tan(fov*math.pi/360))
    fy=fx
    K=np.array([[fx,0,imageWidthHeight[0]/2],[0,fy,imageWidthHeight[1]/2],[0,0,1]])
    # MTrs=np.array([[0,-1,0],[0,0,-1],[-1,0,0]]) 原始的世界坐标到相机坐标没有偏移的变换
    MTrs = np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]])

    # Turn the camera position into a column vector.
    camPosition = np.transpose([camXYZ])

    # Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    pitchRollYaw = utils.to_eularian_angles(camQuaternion)

    # Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = rotation_matrix_from_angles(pitchRollYaw)
    R=np.matmul(MTrs,np.transpose(camRotation)) #说实话不知道这里为什么取逆，反正需要的是把原始的坐标，转换到旋转后的坐标系里

    T=-np.matmul(R,camPosition)

    # Change coordinates to get subjectXYZ in the camera's local coordinate system.
    XYZW = np.transpose([subjectXYZ])
    CamPoint = np.matmul(R,XYZW) + T
    print("cam: ",CamPoint)
    Depth = CamPoint[2]

    ScreenPoint = np.matmul(K,CamPoint)
    ScreenPoint = ScreenPoint/Depth

    # RT1=np.hstack((R,T));
    # RT=np.vstack((RT1,np.array([[0,0,0,1]])))
    # print("RT:",RT)
    # exit()

    if Depth==0:
        return None

    return np.array([
        ScreenPoint[0] ,
        ScreenPoint[1],
        Depth
    ]).reshape(3, )


def get_image(x, y, z, pitch, roll, yaw, client):
    """
    title::
        get_image

    description::
        Capture images (as numpy arrays) from a certain position.

    inputs::
        x
            x position in meters
        y
            y position in meters
        z
            altitude in meters; remember NED, so should be negative to be
            above ground
        pitch
            angle (in radians); in computer vision mode, this is camera angle
        roll
            angle (in radians)
        yaw
            angle (in radians)
        client
            connection to AirSim (e.g., client = MultirotorClient() for UAV)

    returns::
        position
            AirSim position vector (access values with x_val, y_val, z_val)
        angle
            AirSim quaternion ("angles")
        im
            segmentation or IR image, depending upon palette in use (3 bands)
        imScene
            scene image (3 bands)

    author::
        Elizabeth Bondi
        Shital Shah
    """

    # Set pose and sleep after to ensure the pose sticks before capturing image.
    client.simSetVehiclePose(Pose(Vector3r(x, y, z), \
                                  to_quaternion(pitch, roll, yaw)), True)
    time.sleep(0.1)

    # Capture segmentation (IR) and scene images.
    responses = \
        client.simGetImages([ImageRequest("0", ImageType.Infrared,
                                          False, False),
                             ImageRequest("0", ImageType.Scene, \
                                          False, False),
                             ImageRequest("0", ImageType.Segmentation, \
                                          False, False)])

    # Change images into numpy arrays.
    img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
    im = img1d.reshape(responses[0].height, responses[0].width, 4)

    img1dscene = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
    imScene = img1dscene.reshape(responses[1].height, responses[1].width, 4)

    return Vector3r(x, y, z), to_quaternion(pitch, roll, yaw), \
           im[:, :, :3], imScene[:, :, :3]  # get rid of alpha channel


def getMeshVtx(meshName):
    meshes = client.simGetMeshPositionVertexBuffers()
    index=0

    for m in meshes:
        if meshName.lower() == m.name :  #cityenginematerial_m  lot_m_15
            # Convert the lists to numpy arrays #这个可能还是厘米的尺度，和后面的pose有两位数的差异
            vertex_list=np.array(m.vertices,dtype=np.float32)
            print("vertex_list.len ", len(vertex_list))
            np.savetxt("vertex_list"+str(index)+".txt",vertex_list,fmt='%.1f')
            indices=np.array(m.indices,dtype=np.uint32)
            print("indices.len ",len(indices))
            np.savetxt("indices"+str(index)+".txt",indices,fmt='%.000001f')
            index += 1

            num_vertices=int(len(vertex_list)/3)
            num_indices=len(indices)
            vertices_reshaped=vertex_list.reshape((num_vertices,3))
            indices_reshaped=indices.reshape((int(num_indices/3),3))
            MeshVtx = np.hstack((vertices_reshaped, np.ones((vertices_reshaped.shape[0], 1))))
            print(MeshVtx)

            cam_resolusion = (256, 144)
            camera_info = client.simGetCameraInfo(camera_name=0)
            print(camera_info)
            screen_pixels = np.array([[0, 0]])

            #MeshVtx
            for MeshPoint in MeshVtx:
                object_xy_in_pic = project_3d_point_to_screen(
                    [MeshPoint[0]/100, MeshPoint[1]/100, -MeshPoint[2]/100],
                    [camera_info.pose.position.x_val,camera_info.pose.position.y_val, camera_info.pose.position.z_val],
                    camera_info.pose.orientation,
                    camera_info.proj_mat.matrix,
                    [cam_resolusion[0],cam_resolusion[1]]
                )
                screen_pixels=np.concatenate((screen_pixels,object_xy_in_pic.reshape(1,2)),axis=0)
                #print("Object projected to pixel\n{!s}.".format(object_xy_in_pic))

            screen_pixels=np.round(screen_pixels).astype(int)
            print(screen_pixels)
            img = cv2.imread("img.png")
            for p in screen_pixels[1:]:
                cv2.circle(img,p,2,(0,0,255))
            cv2.imshow("img", img)
            cv2.waitKey(0)

