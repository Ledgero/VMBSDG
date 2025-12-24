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

def getRTKByPose(camXYZ, camQuaternion, imageWidthHeight,fov): #这里的转换关系不知道对不对，得验证一下.这里的转换坐标系要着重看一下

    fx=imageWidthHeight[0]/(2*math.tan(fov*math.pi/360))
    fy=fx
    K=np.array([[fx,0,imageWidthHeight[0]/2],[0,fy,imageWidthHeight[1]/2],[0,0,1]])
    MTrs=np.array([[0,-1,0],[0,0,-1],[-1,0,0]])

    # Turn the camera position into a column vector.
    camPosition = np.transpose([camXYZ])

    # Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    pitchRollYaw = utils.to_eularian_angles(camQuaternion)

    # Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = rotation_matrix_from_angles(pitchRollYaw)
    R=np.matmul(MTrs,np.transpose(camRotation))#为什么要trans

    T=-np.matmul(R,camPosition)

    # Change coordinates to get subjectXYZ in the camera's local coordinate system.

    RT1=np.hstack((R,T));
    RT=np.vstack((RT1,np.array([[0,0,0,1]])))

    return RT,K

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
