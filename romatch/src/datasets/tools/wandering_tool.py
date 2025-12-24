# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/main/docs/image_apis.md#computer-vision-mode
import random

#import setup_path
# import airsim
import numpy
import numpy as np
import math
import cv2
import pprint
import tempfile
import os
import time
from romatch.src.datasets.tools import fileProcess
from romatch.src.datasets.tools.mathCompute import getQuaternionToLookAt


# point1=(1,1,100)
# point2=(100,100,100)
#再改一下成为保存到数组里，然后返回数组，直接遍历pos完成位姿设置。
def traverse_rectangleXDirection(point1, point2, step, stepNums_y):
    # 计算矩形的边界
    min_x = min(point1[0], point2[0])
    max_x = max(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    max_y = max(point1[1], point2[1])

    # 初始化起始点和方向
    x = min_x
    y = min_y
    direction_X = 1  # 初始方向向x正方向
    IsXPath = True
    i=0
    datalist=[]

    headDisX = 0
    headDisY = 0
    headDisZ = 0
    headDirectionX = 0
    headDirectionY = 0
    headDirectionZ = 0

    # 遍历矩形内的点
    # 好像结束有问题，但是python自己给他结束了，所以再说
    while x>=min_x and x<=max_x and y>=min_y and y<=max_y:
        # 输出当前点坐标
        if headDisX>=-math.pi/4 and headDisX<=math.pi/4:
            headDisX += math.pi / 16
            headDisY += math.pi / 16
            headDisZ += math.pi / 16

        datalist.append([x,y,headDirectionX+headDisX,headDirectionY+headDisY,headDirectionZ])
        #print(x,y)

        if IsXPath:
            # 更新下一个点的位置
            if direction_X == 1:  # 向x正方向
                if x + step <= max_x:
                    x += step
                    headDirectionZ = 0
                else:
                    y += step
                    headDirectionZ = math.pi / 2
                    direction_X = -1  # 改变方向
                    IsXPath = False
            else:  # x负方向
                if x - step >= min_x:
                    x -= step
                    headDirectionZ = math.pi
                else: #每次else的时候都是换方向的时候
                    y += step
                    headDirectionZ = math.pi/2
                    direction_X = 1  # 改变方向向右
                    IsXPath = False
        else: #当前点要更新的是Y方向了
            if y + step <= max_y and i < stepNums_y:
                y += step
                i += 1
                headDirectionZ = math.pi/2
            else:
                i=0   #清零
                IsXPath = True #恢复X 方向前进
                #这里会多加一次当前的（x , y），所以需要先减掉一个
                datalist.pop()
    return np.array(datalist)

def getHeadDirection():
    # Diections=[[0,0,0],[math.pi/4,0,0],[0,math.pi/4,0],[0,0,math.pi/4],
    #            [-math.pi/4,0,0],[0,-math.pi/4,0],[0,0,-math.pi/4],
    #
    #            [math.pi/7,0,math.pi/7],[math.pi/7,math.pi/7,0],[0,math.pi/7,math.pi/7],[math.pi/7,math.pi/7,math.pi/7],
    #
    #            [-math.pi / 7, 0, -math.pi / 7], [-math.pi / 7, -math.pi / 7, 0], [0, -math.pi / 7, -math.pi / 7],
    #             [-math.pi / 7, 0, math.pi / 7], [-math.pi / 7, math.pi / 7, 0], [0, -math.pi / 7, math.pi / 7],
    #            [math.pi / 7, 0, -math.pi / 7], [math.pi / 7, -math.pi / 7, 0], [0, math.pi / 7, -math.pi / 7],
    #
    #            [-math.pi / 7, -math.pi / 7, -math.pi / 7], [math.pi / 7, -math.pi / 7, -math.pi / 7],
    #            [-math.pi / 7, math.pi / 7, -math.pi / 7],[-math.pi / 7, -math.pi / 7, math.pi / 7]]

    Diections = [[0, 0, 0], [math.pi / 7, 0, 0], [0, math.pi / 7, 0], [0, 0, math.pi / 7],
                 [-math.pi / 7, 0, 0], [0, -math.pi / 7, 0], [0, 0, -math.pi / 7]]
    return numpy.array(Diections)

#stepZ可以考虑40，遍历的点位是按照step均匀采集点位，然四个方向角度为主方向，主方向与随机偏移方向凑成一对（同时根据高度搞一个仰头的角度）
def traverse_AllRectangle(point1, point2, step, radius):
    min_x = min(point1[0], point2[0])
    max_x = max(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    max_y = max(point1[1], point2[1])

    Zlist=[0,-40,-80]
    angles=[-90,0,90,180]

    datalist = []
    # datalist1 = []

    for x in range(min_x,max_x+1,step):
        for y in range(min_y, max_y+1, step):
            angle=random.uniform(0,360)
            rad=math.radians(angle)
            newX = x + radius * math.cos(rad) #这里的距离还可以变
            newY = y + radius * math.sin(rad) #rad可能是与X轴正方向夹角，往Y轴递增
            # for z in range (min_z, max_z+1, stepZ):
            for z in Zlist:
                for angleA in angles:
                    angleOffset0 = random.uniform(0, 30)
                    angleOffset1 = random.uniform(0, 20)
                    angleB_Z = angleA + angleOffset0
                    if z < -60:
                        angleB_Y = - angleOffset1
                    else:
                        angleB_Y = angleOffset1
                    datalist.append((x,y,z,0,0,math.radians(angleA),  newX,newY,z,math.radians(angleB_Y),0,math.radians(angleB_Z))) #角度应该不会对效果有很大的影响
    print(len(datalist))
    return np.array(datalist)

#采集z轴范围更大，相比于traverse_AllRectangle
def traverse_AllRectangle1(point1, point2, step, radius_base):
    min_x = min(point1[0], point2[0])
    max_x = max(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    max_y = max(point1[1], point2[1])

    Zlist=[0,-20,-50,-80]
    angles=[-90,0,90,180]

    datalist = []
    # datalist1 = []

    for x in range(min_x,max_x+1,step):
        for y in range(min_y, max_y+1, step):
            # for idd in range(3):#同一个地点循环三次随机位置
                angle=random.uniform(0,360)
                rad=math.radians(angle)
                radius=random.uniform(radius_base-15,radius_base+15) #采样间距变化
                newX = x + radius * math.cos(rad) #这里的距离还可以变
                newY = y + radius * math.sin(rad) #rad可能是与X轴正方向夹角，往Y轴递增
                # for z in range (min_z, max_z+1, stepZ):
                #for -100 to 0 处理三次
                for z in Zlist:
                    for angleA in angles:
                        angleB_Z = angleA + random.uniform(0, 30)
                        if z < -60:
                            angleB_Y = - random.uniform(0, 30) #低头
                        else:
                            angleB_Y = random.uniform(0, 20) #仰头
                        datalist.append((x,y,z,0,0,math.radians(angleA),  newX,newY,z,math.radians(angleB_Y),0,math.radians(angleB_Z))) #角度应该不会对效果有很大的影响
    print(len(datalist))
    return np.array(datalist)

#for city02 z:[-100,-10]
def traverse_AllRectangle2(point1, point2, step, radius):
    min_x = min(point1[0], point2[0])
    max_x = max(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    max_y = max(point1[1], point2[1])

    Zlist=[120,110,100,90,70,20,-40] #疑问，city02，Z不应该是负值才越高吗
    angles=[-90,0,90,180]

    datalist = []
    # datalist1 = []

    for x in range(min_x,max_x+1,step):
        for y in range(min_y, max_y+1, step):
            angle=random.uniform(0,360)
            rad=math.radians(angle)
            newX = x + radius * math.cos(rad) #这里的距离还可以变
            newY = y + radius * math.sin(rad) #rad可能是与X轴正方向夹角，往Y轴递增
            # for z in range (min_z, max_z+1, stepZ):
            for z in Zlist:
                for angleA in angles:
                    angleOffset0 = random.uniform(0, 30)
                    angleOffset1 = random.uniform(0, 20)
                    angleB_Z = angleA + angleOffset0
                    if z < 0:
                        angleB_Y = - angleOffset1
                    else:
                        angleB_Y = angleOffset1
                    datalist.append((x,y,z,0,0,math.radians(angleA),  newX,newY,z,math.radians(angleB_Y),0,math.radians(angleB_Z))) #角度应该不会对效果有很大的影响
    print(len(datalist))
    return np.array(datalist)

#radius base一开始小一点
def traverse_Zoom(point1, point2, step, radius_base, maxZoomDis):
    start_time = time.time()
    min_x = min(point1[0], point2[0])
    max_x = max(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    max_y = max(point1[1], point2[1])

    Zlist = [0,-60]
    betas=[0,60,120,180,240,300]
    alphsZ0=[0,50]
    alphsZHigh = [0, -30, -50]#相当于水平线朝下30度 50度
    zoomScale = [1/4,1/2,2/3]


    datalist = []
    # datalist1 = []
    for x in range(min_x,max_x+1,step):
        for y in range(min_y, max_y+1, step):
            for z in Zlist:
                # x=-303
                # y=-253
                # z=-35 test
                alphs = alphsZ0 if z==0 else alphsZHigh
                for alph in alphs:
                    for beta in betas:
                        LookAtPosX = x + maxZoomDis * math.cos(math.radians(beta + random.uniform(-30, 30)))
                        LookAtPosY = y + maxZoomDis * math.sin(math.radians(beta + random.uniform(-30, 30)))
                        LookAtPosZ = z - maxZoomDis * math.sin(math.radians(alph))
                        LookatPos = np.array([LookAtPosX, LookAtPosY, LookAtPosZ])
                        # LookatPos = np.array([-200.79007762, -150.26747884,  118.20888862]) test

                        # rad = math.radians(random.uniform(0, 360))
                        # radius = random.uniform(radius_base - 5, radius_base + 5)  # 采样间距变化
                        # newX = x + radius * math.cos(rad)  # 这里的距离还可以变
                        # newY = y + radius * math.sin(rad)  # rad可能是与X轴正方向夹角，往Y轴递增
                        for scale in zoomScale:
                            #计算四元数视角
                            OrigPos0=[x,y,z]
                            OrigPos1=[x,y,z]
                            OrigPos1ZoomNp=scale*np.array(LookatPos)+(1-scale)*np.array(OrigPos1)
                            OrigPos1Zoom=[OrigPos1ZoomNp[0],OrigPos1ZoomNp[1],OrigPos1ZoomNp[2]]

                            Q0=getQuaternionToLookAt(OrigPos0,LookatPos);
                            datalist.append((OrigPos0,OrigPos1Zoom,Q0,Q0)) #角度应该不会对效果有很大的影响
    print("spend: ",time.time()-start_time)
    print(len(datalist))
    return np.array(datalist,dtype=object)

def traverse_Test(point1, point2, step, radius): #stepZ可以考虑40
    min_x = min(point1[0], point2[0])
    max_x = max(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    max_y = max(point1[1], point2[1])
    datalist = []

    for x in range(min_x,max_x+1,step):
        for y in range(min_y, max_y+1, step):
            x=0
            y=0
            angle=random.uniform(0,360)
            rad=math.radians(angle)
            newX = x + radius * math.cos(rad)
            newY = y + radius * math.sin(rad) #rad可能是与X轴正方向夹角，往Y轴递增
            # for z in range (min_z, max_z+1, stepZ):
            for z in [0,-40,-80]:
                for angleA in [-90,0,90,180]:
                    angleOffset0 = random.uniform(0, 30)
                    angleOffset1 = random.uniform(0, 20)
                    angleB_Z = angleA + angleOffset0
                    if z < -60:
                        angleB_Y = - angleOffset1
                    else:
                        angleB_Y = angleOffset1
                    datalist.append((x,y,z,0,0,math.radians(angleA),  newX,newY,z,math.radians(angleB_Y),0,math.radians(angleB_Z))) #角度应该不会对效果有很大的影响
    print(len(datalist))
    return np.array(datalist)

def remap(img, mapX, mapY): #可能的插值方式，先不动，看看相互映射的点对不对，是不是相互映射像素都一样,先不管插值？或者自己弄一个插值
    #不过我的就是用来筛选的 可以先不做
    MapX = mapX.astype(np.int32)
    MapY = mapY.astype(np.int32)
    map=np.stack([MapX,MapY],axis=0)
    img2=img[map[1, :, :],map[0, :, :]] #这个的意思是 img2对应位置的像素值，是img中map存储的像素值，map里存储的要是img的像素对应

    print("test buf UV：",[map[1, 132, 442],map[0, 132, 442]])
    return img2

# # def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
# def warp_img():
#     mapX=np.array([[0,1,2],[0,1,2],[0,1,2]])
#     mapY=np.array([[2,2,2],[1,1,1],[0,0,0]])
#      #不在范围内筛选一下，统一移动到一个像素中，或者用gpt写一下
#     var = remap(depth0,mapX,mapY )
#     print(var)


# 输入K0, K1 ,T0to1, depth0, depth1, cam_resolusion(img0的分辨率)， 输出overlap（重叠系数）, MapX, MapY（X Y坐标的变换映射，一般是做图片变换用的,img1变到img0）
def findMap(K0, K1 ,T0to1, depth0, depth1, cam_resolusion): # M=K1`T0to1 * K0`^-1
    K0_buf=np.vstack((K0,np.array([[0,0,0]])))
    K0_add=np.hstack((K0_buf,np.array([[0],[0],[0],[1]])))
    K1_buf = np.vstack((K1, np.array([[0, 0, 0]])))
    K1_add = np.hstack((K1_buf, np.array([[0], [0], [0], [1]])))
    M=np.matmul(np.matmul(K1_add,T0to1),np.linalg.inv(K0_add))

    X = np.arange(cam_resolusion[0])
    Y = np.arange(cam_resolusion[1])
    one = np.ones((cam_resolusion[1], cam_resolusion[0]))
    Xpos, Ypos = np.meshgrid(X, Y)
    rePos = np.stack([depth0 * Xpos, depth0 * Ypos, depth0 * one, one], axis=0)
    AnsBuf = np.einsum('ij,jmn->imn', M, rePos)

    #筛选超过分辨率
    Ans = AnsBuf/(AnsBuf[2]+1e-6) 
    Ans[0]=np.where((Ans[0]>=cam_resolusion[0])| (Ans[0]<0),0,Ans[0])
    Ans[1]=np.where((Ans[1]>=cam_resolusion[1])| (Ans[1]<0),0,Ans[1])
    #筛选深度差距过大的,以及背景是0的筛除掉，合成数据是小于-5000的筛选掉
    # Ans[0]=np.where((abs(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]-AnsBuf[2])>5.0)|(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]<-5000) ,0,Ans[0]) #深度差距过大筛掉，映射到（0 ，0）
    # Ans[1]=np.where((abs(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]-AnsBuf[2])>5.0)|(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]<-5000) ,0,Ans[1]) #但是可能也不是完全能对的上，应该有些误差，稍稍放宽一点
    # #这里可做一遍检查，筛选矫正那些depth不对的情况，后面再说
    Ans[0]=np.where((abs(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]-AnsBuf[2])>5.0) | (abs(depth0)<1e-6)|(depth0<-5000),0,Ans[0]) #深度差距过大筛掉，映射到（0 ，0）
    Ans[1]=np.where((abs(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]-AnsBuf[2])>5.0) | (abs(depth0)<1e-6)|(depth0<-5000),0,Ans[1]) #但是可能也不是完全能对的上，应该有些误差，稍稍放宽一点
    
    UV0to1_unfilter = Ans.transpose(2, 1, 0).reshape((-1, 4), order='C')#可能是U，V，1，1/depth
    #筛一些有效像素。统计在1上有多少像素。
    mask = (UV0to1_unfilter[:, 0] < cam_resolusion[0]) & (UV0to1_unfilter[:, 0] > 0) & (UV0to1_unfilter[:, 1] < cam_resolusion[1]) & (UV0to1_unfilter[:, 1] > 0)
    overlap=mask.sum()/mask.shape[0]

    MapY = np.where(Ans[0] == 0, 0, Ans[1]).astype(np.float32) #只要有X一个坐标为0,对应的Y置0
    MapX = np.where(Ans[1] == 0, 0, Ans[0]).astype(np.float32)
    # UV0to1_Reshape=UV0to1_unfilter[mask]

    return overlap, MapX, MapY
    #MapX 0-960 #MapY 0-540 MapX Y标识的是对应位置上的对应像素。
    #本身位置是（像素位置是img0X img0Y）, MapXY中存储的是当前像素对应的img1X img1Y
    #那么映射时，传入img1，直接把MapXY 对应像素img1(MapX MapY),放置到本身位置上，也就是img0的对应位置（实现img1到 img0的映射）
    #注意T转换矩阵是T0to1 img0的点坐标转换到img1上是多少，得到MapXY，但是投影是img1 到 0

def findMapAndGetOverlap(K0, K1 ,T0, T1, depth0, depth1, cam_resolusion, imagePath='', debugPic = False): # 这是获得整体的overlap，相当于双向计算并平均
    T_0to1 = np.matmul(T1, np.linalg.inv(T0))
    T_1to0 = np.matmul(T0, np.linalg.inv(T1))

    overlap1T0, MapX1T0, MapY1T0 = findMap(K0, K1, T_0to1, depth0, depth1, cam_resolusion)
    overlap0T1, MapX0T1, MapY0T1 = findMap(K1, K0, T_1to0, depth1, depth0, cam_resolusion)
    print("overlap1T0 ",overlap1T0)
    print("overlap0T1 ", overlap0T1)

    if(debugPic):
        src1 = cv2.imread(os.path.join(imagePath, 'img1.png'), cv2.IMREAD_COLOR)
        img1_t0 = cv2.remap(src1, MapX1T0, MapY1T0, cv2.INTER_LINEAR)
        fileProcess.numpy_to_file(os.path.normpath(os.path.join(imagePath, 'img_trans1T0.png')), img1_t0) #img1To0

        src0 = cv2.imread(os.path.join(imagePath, 'img0.png'), cv2.IMREAD_COLOR)
        img0_t1 = cv2.remap(src0, MapX0T1, MapY0T1, cv2.INTER_LINEAR)
        fileProcess.numpy_to_file(os.path.normpath(os.path.join(imagePath, 'img_trans0T1.png')), img0_t1) #img0To1

    # return (overlap1T0 + overlap0T1)/2.0
    return max(overlap1T0 , overlap0T1)

# 输入K0, K1 ,T0to1, depth0, depth1, cam_resolusion(img0的分辨率)， 输出overlap（重叠系数）, MapX, MapY（X Y坐标的变换映射，一般是做图片变换用的,img1变到img0）
def findMapAndGetBoundBox(K0, K1 ,T0to1, depth0, depth1, cam_resolusion): # M=K1`T0to1 * K0`^-1
    K0_buf=np.vstack((K0,np.array([[0,0,0]])))
    K0_add=np.hstack((K0_buf,np.array([[0],[0],[0],[1]])))
    K1_buf = np.vstack((K1, np.array([[0, 0, 0]])))
    K1_add = np.hstack((K1_buf, np.array([[0], [0], [0], [1]])))
    M=np.matmul(np.matmul(K1_add,T0to1),np.linalg.inv(K0_add))

    X = np.arange(cam_resolusion[0])
    Y = np.arange(cam_resolusion[1])
    one = np.ones((cam_resolusion[1], cam_resolusion[0]))
    Xpos, Ypos = np.meshgrid(X, Y)
    rePos = np.stack([depth0 * Xpos, depth0 * Ypos, depth0 * one, one], axis=0)
    AnsBuf = np.einsum('ij,jmn->imn', M, rePos)

    #筛选超过分辨率
    Ans = AnsBuf/(AnsBuf[2]+1e-6) 
    Ans[0]=np.where((Ans[0]>=cam_resolusion[0])| (Ans[0]<0),0,Ans[0])
    Ans[1]=np.where((Ans[1]>=cam_resolusion[1])| (Ans[1]<0),0,Ans[1])
    #筛选深度差距过大的,以及背景是0的筛除掉，合成数据是小于-5000的筛选掉
    # Ans[0]=np.where((abs(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]-AnsBuf[2])>5.0)|(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]<-5000) ,0,Ans[0]) #深度差距过大筛掉，映射到（0 ，0）
    # Ans[1]=np.where((abs(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]-AnsBuf[2])>5.0)|(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]<-5000) ,0,Ans[1]) #但是可能也不是完全能对的上，应该有些误差，稍稍放宽一点
    # #这里可做一遍检查，筛选矫正那些depth不对的情况，后面再说
    Ans[0]=np.where((abs(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]-AnsBuf[2])>5.0) | (abs(depth0)<1e-6)|(depth0<-5000),0,Ans[0]) #深度差距过大筛掉，映射到（0 ，0）
    Ans[1]=np.where((abs(depth1[Ans[1].astype(np.int32), Ans[0].astype(np.int32)]-AnsBuf[2])>5.0) | (abs(depth0)<1e-6)|(depth0<-5000),0,Ans[1]) #但是可能也不是完全能对的上，应该有些误差，稍稍放宽一点
    
    UV0to1_unfilter = Ans.transpose(2, 1, 0).reshape((-1, 4), order='C')#可能是U，V，1，1/depth
    #筛一些有效像素。统计img0投射到1上有效的，img0的像素个数
    mask = (UV0to1_unfilter[:, 0] < cam_resolusion[0]) & (UV0to1_unfilter[:, 0] > 0) & (UV0to1_unfilter[:, 1] < cam_resolusion[1]) & (UV0to1_unfilter[:, 1] > 0)
    overlap=mask.sum()/mask.shape[0]

    MapY = np.where(Ans[0] == 0, 0, Ans[1]).astype(np.float32) #只要有X一个坐标为0,对应的Y置0
    MapX = np.where(Ans[1] == 0, 0, Ans[0]).astype(np.float32)
    # UV0to1_Reshape=UV0to1_unfilter[mask]
    minP,maxP=np.min(np.where(UV0to1_unfilter==0,10000000,UV0to1_unfilter),axis=0)[:2], np.max(UV0to1_unfilter,axis=0)[:2] #重叠区域boundingbox

    return overlap, MapX, MapY, minP, maxP
    #MapX 0-960 #MapY 0-540 MapX Y标识的是对应位置上的对应像素。
    #本身位置是（像素位置是img0X img0Y）, MapXY中存储的是当前像素对应的img1X img1Y
    #那么映射时，传入img1，直接把MapXY 对应像素img1(MapX MapY),放置到本身位置上，也就是img0的对应位置（实现img1到 img0的映射）
    #注意T转换矩阵是T0to1 img0的点坐标转换到img1上是多少，得到MapXY，但是投影是img1 到 0

#findMapAndGetOverlap不同返回数版本
def GetOverlapRtnBoth(K0, K1 ,T0, T1, depth0, depth1, cam_resolusion, imagePath='', debugPic = False): # 这是获得整体的overlap，相当于双向计算并平均
    T_0to1 = np.matmul(T1, np.linalg.inv(T0))
    T_1to0 = np.matmul(T0, np.linalg.inv(T1))

    overlap1T0, MapX1T0, MapY1T0 = findMap(K0, K1, T_0to1, depth0, depth1, cam_resolusion)
    overlap0T1, MapX0T1, MapY0T1 = findMap(K1, K0, T_1to0, depth1, depth0, cam_resolusion)
    print("overlap1T0 ",overlap1T0)
    print("overlap0T1 ", overlap0T1)

    if(debugPic):
        src1 = cv2.imread(os.path.join(imagePath, 'img1.png'), cv2.IMREAD_COLOR)
        img1_t0 = cv2.remap(src1, MapX1T0, MapY1T0, cv2.INTER_LINEAR)
        fileProcess.numpy_to_file(os.path.normpath(os.path.join(imagePath, 'img_trans1T0.png')), img1_t0) #img1To0

        src0 = cv2.imread(os.path.join(imagePath, 'img0.png'), cv2.IMREAD_COLOR)
        img0_t1 = cv2.remap(src0, MapX0T1, MapY0T1, cv2.INTER_LINEAR)
        fileProcess.numpy_to_file(os.path.normpath(os.path.join(imagePath, 'img_trans0T1.png')), img0_t1) #img0To1

    # return (overlap1T0 + overlap0T1)/2.0
    return overlap1T0 , overlap0T1

# path0=traverse_AllRectangle((289, -253), (-303, 244),40,20) #AB相距20 step/2
# print(path0.size)
# a=traverse_rectangleXDirection(point1,point2,10,1)
# print(a)

#输入两个尺寸不一样的图像，做comb 投影图来check.注意都是从左上角（0，0）开始的图
#img0,img1 size不一样。depth0,depth1 需要按照分辨率截取
#假设都是numpy类型 img0 img1是灰度图
def getCheckImgComb(img0,img1,depth0,depth1,K0,K1,T_0to1,T_1to0):

    #其实要做各种类型检查
    cam_resolusion = (max(img0.shape[1],img1.shape[1]),max(img0.shape[0],img1.shape[0]))
    #大小不一样需要做填充
    padH0, padW0 = int(cam_resolusion[1]-img0.shape[0]), int(cam_resolusion[0]-img0.shape[1])
    padH1, padW1 = int(cam_resolusion[1]-img1.shape[0]), int(cam_resolusion[0]-img1.shape[1])

    img0=np.pad(img0,((0,padH0),(0,padW0)),'constant',constant_values=0)
    img1=np.pad(img1,((0,padH1),(0,padW1)),'constant',constant_values=0)

    depth0=depth0[0:cam_resolusion[1], 0:cam_resolusion[0]]
    depth1=depth1[0:cam_resolusion[1], 0:cam_resolusion[0]]
    overlap_buf, MapX0, MapY0 = findMap(K0, K1, T_0to1, depth0, depth1, cam_resolusion)

    img1_t0 = cv2.remap(img1, MapX0, MapY0, cv2.INTER_LINEAR)
    overlap_buf1, MapX1, MapY1 = findMap(K1, K0, T_1to0, depth1, depth0, cam_resolusion)
    img0_t1 = cv2.remap(img0, MapX1, MapY1, cv2.INTER_LINEAR)
    imgTransCo = np.hstack((img1_t0, img0_t1))
    imgComb = np.hstack((img0, img1))
    imgAll = np.vstack((imgComb, imgTransCo))

    return imgAll

#服务于找左上角第一个非0元素（for mapXY）
def nearest_non_zero(matrix):
    # 找到所有非零元素的索引
    non_zero_indices = np.transpose(np.nonzero(matrix))

    # 计算每个非零元素到左上角 (0, 0) 的距离
    distances = np.sqrt(np.sum(non_zero_indices ** 2, axis=1))

    # 找到距离最小的非零元素的索引
    nearest_index = np.argmin(distances)
    nearest_value = matrix[non_zero_indices[nearest_index][0], non_zero_indices[nearest_index][1]]

    return nearest_value, tuple(non_zero_indices[nearest_index])