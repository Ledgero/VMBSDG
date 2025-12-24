import os
import sys
import tools.wandering_tool
sys.path.append("..//data")
import airsim
from tools import wandering_tool
from tools import fileProcess
from tools import pointProjector_debug
import numpy as np
import h5py
from tools import pose2RTK
import pprint
import time
import cv2
from tools import pointProjector as ppr
from tools import pointProjector_debug as ppr_debug

def project3DtoScreenBy2Ways(): #这个函数验证了project_3d_point_to_screen与project_3d_point_to_screen_By_RTK有一样的效果。
                                #也就是说project_3d_point_to_screen_By_RTK计算方式是对的。
    client = airsim.VehicleClient()
    client.confirmConnection()
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, -1.57079633)), True)

    cam_resolusion = (960, 540)
    MeshPoint = np.array([-6.309, -80.7614, -23.477])
    # MeshPoint = [647.5, 8193.7, 2351.7];
    camera_info = client.simGetCameraInfo(camera_name=0)
    print(camera_info)

    object_xy_in_pic0 = ppr.project_3d_point_to_screen(
        [MeshPoint[0] , MeshPoint[1] , MeshPoint[2] ],
        [camera_info.pose.position.x_val, camera_info.pose.position.y_val, camera_info.pose.position.z_val],
        camera_info.pose.orientation,
        camera_info.proj_mat.matrix,
        [cam_resolusion[0], cam_resolusion[1]]
    )
    object_xy_in_pic1 = ppr.project_3d_point_to_screen_By_RTK(
        [MeshPoint[0] , MeshPoint[1], MeshPoint[2]],
        [camera_info.pose.position.x_val, camera_info.pose.position.y_val, camera_info.pose.position.z_val],
        camera_info.pose.orientation,
        camera_info.proj_mat.matrix,
        [cam_resolusion[0], cam_resolusion[1]],
        89.9
    )
    print("object_xy_in_pic0",object_xy_in_pic0)
    print("object_xy_in_pic1",object_xy_in_pic1)

def picsPointMatchCheck():
    client = airsim.VehicleClient()
    client.confirmConnection()
    cam_resolusion = (960, 540)

    client.simSetVehiclePose(
        airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, -1.57079633)), True)
    responses1 = client.simGetImages([  # 可能需要小小sleep一下消除差异，之后测试一下。看是不是不同步？？
        airsim.ImageRequest("0", airsim.ImageType.Scene),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)])
    for i, response in enumerate(responses1):
        if response.pixels_as_float:
            img_depth_planar1 = np.array(response.image_data_float).reshape(response.height, response.width)
        else:
            curImg1 = fileProcess.bytes_to_numpy(response.image_data_uint8)
            imagePath="D:\\Users\\forev\\PycharmProjects\\AirSim_Mountain\\调试"
            airsim.write_file(os.path.normpath(os.path.join(imagePath, 'img1.png')),response.image_data_uint8)
    CamInfo1 = client.simGetCameraInfo(camera_name=0)
    pose1, K1 = pose2RTK.getRTKByPose(
        [CamInfo1.pose.position.x_val, CamInfo1.pose.position.y_val, CamInfo1.pose.position.z_val],
        CamInfo1.pose.orientation, [cam_resolusion[0], cam_resolusion[1]], CamInfo1.fov)


    client.simSetVehiclePose(
        airsim.Pose(airsim.Vector3r(-2.87055852, -19.79292535, 0), airsim.to_quaternion(0.04582258, 0, -1.11597639)), True)
    responses2 = client.simGetImages([  # 可能需要小小sleep一下消除差异，之后测试一下。看是不是不同步？？
        airsim.ImageRequest("0", airsim.ImageType.Scene),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)])
    for i, response in enumerate(responses2):
        if response.pixels_as_float:
            img_depth_planar2 = np.array(response.image_data_float).reshape(response.height, response.width)
        else:
            curImg2 = fileProcess.bytes_to_numpy(response.image_data_uint8)
            imagePath="D:\\Users\\forev\\PycharmProjects\\AirSim_Mountain\\调试"
            airsim.write_file(os.path.normpath(os.path.join(imagePath, 'img2.png')),response.image_data_uint8)
    CamInfo2 = client.simGetCameraInfo(camera_name=0)
    pose2, K2 = pose2RTK.getRTKByPose(
        [CamInfo2.pose.position.x_val, CamInfo2.pose.position.y_val, CamInfo2.pose.position.z_val],
        CamInfo2.pose.orientation, [cam_resolusion[0], cam_resolusion[1]], CamInfo2.fov)

    UV1 = np.array([[442], [132]])
    UV2 = np.array([[216], [86]])

    print(img_depth_planar1[UV1[1][0]][UV1[0][0]])
    pointProjector_debug.check_CityDataSet(K1, K2, pose1, pose2, img_depth_planar1, img_depth_planar2, UV1, UV2)

#picsPointMatchCheck()
project3DtoScreenBy2Ways()