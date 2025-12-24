import sys
sys.path.append("C:\\Users\\forev\\.conda\\envs\\ASMountain\\Lib\\site-packages\\airsim")
import numpy as np
import math
# import airsim


#观察朝向转偏航角（全局坐标系），y x z, 给定一个三维向量，计算右手法则旋转，按照y x z的顺序输出
def computeTheta(normal):
    X = np.array([1, 0, 0])
    Y = np.array([0, 1, 0])
    Z = np.array([0, 0, 1])
    normalX=np.array([0,normal[1],normal[2]]) #x归零，投影到yoz平面，下同理
    normalY = np.array([normal[0], 0, normal[2]])
    normalZ = np.array([normal[0], normal[1], 0])
    data_M = np.sqrt(np.sum(normal * normal))
    data_N = 1

    #绕X轴旋转角度
    cos_thetaX = np.sum(normalX * Z) / (np.sqrt(np.sum(normalX * normalX)) * data_N)#从Z开始计算角度
    cos_thetaXJudge = np.sum(normalX * Y) / (np.sqrt(np.sum(normalX * normalX)) * data_N)
    thetaX = np.arccos(cos_thetaX) if np.arccos(cos_thetaXJudge)<np.pi/2.0 else -np.arccos(cos_thetaX)
    #绕Y轴旋转角度
    cos_thetaY = np.sum(normalY * X) / (np.sqrt(np.sum(normalY * normalY)) * data_N)
    cos_thetaYJudge = np.sum(normalY * Z) / (np.sqrt(np.sum(normalY * normalY)) * data_N)
    thetaY = np.arccos(cos_thetaY) if np.arccos(cos_thetaYJudge) < np.pi / 2.0 else -np.arccos(cos_thetaY)
    #绕Z轴旋转角度
    cos_thetaZ = np.sum(normalZ * X) / (np.sqrt(np.sum(normalZ * normalZ)) * data_N)
    cos_thetaZJudge = np.sum(normalZ * Y) / (np.sqrt(np.sum(normalZ * normalZ)) * data_N)
    thetaZ = np.arccos(cos_thetaZ) if np.arccos(cos_thetaZJudge) < np.pi / 2.0 else -np.arccos(cos_thetaZ)
    print(np.degrees(thetaY))
    print(np.degrees(thetaX))
    print(np.degrees(thetaZ))
    return thetaY, thetaX, thetaZ

#纯粹的把fromVector转换成toVector，返回四元数.这里可能可以直接算
def getQuaternion(fromVector, toVector):
    fromVector = np.array(fromVector)
    fromVector_e = fromVector / np.linalg.norm(fromVector)

    toVector = np.array(toVector)
    toVector_e = toVector / np.linalg.norm(toVector)

    cross = np.cross(fromVector_e, toVector_e)

    cross_e = cross / np.linalg.norm(cross)

    dot = np.dot(fromVector_e, toVector_e)

    angle = math.acos(dot)

    if angle == 0 or angle == math.pi:
        return False
    else:
        q = airsim.Quaternionr()
        q.w_val = math.cos(angle / 2)
        q.x_val = cross_e[0] * math.sin(angle / 2)
        q.y_val = cross_e[1] * math.sin(angle / 2)
        q.z_val = cross_e[2] * math.sin(angle / 2)
        return q
        # return [ cross_e[0] * math.sin(angle / 2),cross_e[1] * math.sin(angle / 2),
        #          cross_e[2] * math.sin(angle / 2),math.cos(angle / 2)]

#调整Y的姿势
def getQuaternionTransYHoriz(YAxis, LookAtX):
    newToVector=np.array([-LookAtX[1],LookAtX[0],0])
    return getQuaternion(YAxis, newToVector)

def getQuaternionToLookAt(OrigPos_p, LookAt_p):
    LookAtP = np.array(LookAt_p)
    OrigPosP = np.array(OrigPos_p)
    XVec = np.array([1, 0, 0])
    LookAtX = LookAtP - OrigPosP
    AnsReturn0 = getQuaternion(XVec, LookAtX)

    YVecQ = airsim.Quaternionr(0, 1, 0, 0).rotate(AnsReturn0)
    YVec=np.array([YVecQ.x_val, YVecQ.y_val, YVecQ.z_val])

    newToVector = np.array([-LookAtX[1], LookAtX[0], 0])

    cosAngle=np.dot(YVec,newToVector)/(np.linalg.norm(YVec)*np.linalg.norm(newToVector))
    Direction = np.cross(YVec,newToVector)
    if cosAngle>1.0 and cosAngle-1.0<0.0001:
        cosAngle=1.0
    rad = np.arccos(cosAngle)

    #再做调整，保证Y水平，Z总体是向上的--Z总体向上的保证是从YVecQ是投影到地面水平平面来的
    if np.dot(Direction,LookAtX)>0:
        q0 = airsim.to_quaternion(0, rad, 0)
        ans = AnsReturn0.__mul__(q0)
    elif np.dot(Direction,LookAtX)<0:
        q0 = airsim.to_quaternion(0, 2*np.pi-rad, 0)
        ans = AnsReturn0.__mul__(q0)
    elif abs(np.linalg.norm(Direction))<0.0000001: #其实合并到上面也一样
        ans = AnsReturn0
    #测试完成后底下的可以删掉
    #toVQ = YVecQ.rotate(q0)
    # q1 = airsim.to_quaternion(0, -rad, 0)
    # print("toVQ.z_val: ",toVQ.z_val)
    # ans = AnsReturn0.__mul__(q0)
    # #这里可能有问题，好像是不需要q1
    # if abs(toVQ.z_val)<0.00001:
    #     print("0")
    #     ans = AnsReturn0.__mul__(q0)
    # else:
    #     print("1")
    #     ans = AnsReturn0.__mul__(q1)
    return ans