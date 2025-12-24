import os
import h5py
import numpy as np
import cv2

def writeH5(depthPath,x,img_depth_planar):
    file = h5py.File(os.path.normpath(os.path.join(depthPath, 'img' + str(x) + '.h5')), "w")
    file.create_dataset(name='depth', data=img_depth_planar)
    file.close()

def writeH5(depthPath,img_depth_planar):
    file = h5py.File(depthPath, "w")
    file.create_dataset(name='depth', data=img_depth_planar)
    file.close()

# bytes转数组
def bytes_to_numpy(image_bytes):
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2

# 数组保存
def numpy_to_file(pathName,image_np):
    cv2.imwrite(pathName,image_np)
    print(pathName)
    return pathName