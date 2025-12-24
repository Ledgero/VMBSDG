import numpy as np
import cv2
import os.path as osp
# import metrics.estimate_pose, error_auc, relative_pose_error
import metrics

R_errs=[]
t_errs=[]

def SIFTPoseErro(img1,img2,K1,K2,T1to2):
    detector = cv2.SIFT_create()

    kp1 = detector.detect(img1,None)
    kp2 = detector.detect(img2,None)
    kp1,des1 = detector.compute(img1,kp1)
    kp2,des2 = detector.compute(img2,kp2)
    # print(len(kp1),len(des1))
    # print(kp1[0],des1[0])
    # exit()

    # 创建FLANN matcher对象
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 存储好的匹配点
    good_matches = []
    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    # 提取匹配点的坐标
    for dmatch in enumerate(matches):
        if len(dmatch[1])!=2:
            continue

        i=dmatch[0]
        m=dmatch[1][0]
        n=dmatch[1][1]
        
        pts1[i] = kp1[m.queryIdx].pt
        pts2[i] = kp2[m.trainIdx].pt
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # 选取四对点
    pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 估计变换矩阵
    ret = metrics.estimate_pose(pts1, pts2, K1, K2, thresh=0.5, conf=0.99999)
    if ret is None:
        print("none")
    else:
        R, t, inliers = ret
        t_err, R_err = metrics.relative_pose_error(T1to2, R, t, ignore_gt_t_thr=0.00001)
        R_errs.append(R_err)
        t_errs.append(t_err)
        print(t_err, R_err)

def ORBPoseErro(img1,img2,K1,K2,T1to2):
    detector = cv2.ORB_create()
    # detector = cv2.SIFT_create()

    kp1 = detector.detect(img1,None)
    kp2 = detector.detect(img2,None)
    kp1,des1 = detector.compute(img1,kp1)
    kp2,des2 = detector.compute(img2,kp2)

    # 创建FLANN matcher对象
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,     # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 存储好的匹配点
    good_matches = []
    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    # 提取匹配点的坐标
    for dmatch in enumerate(matches):
        if len(dmatch[1])!=2:
            continue

        i=dmatch[0]
        m=dmatch[1][0]
        n=dmatch[1][1]
        
        pts1[i] = kp1[m.queryIdx].pt
        pts2[i] = kp2[m.trainIdx].pt
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # 选取四对点
    pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 估计变换矩阵
    ret = metrics.estimate_pose(pts1, pts2, K1, K2, thresh=0.5, conf=0.99999)
    if ret is None:
        print("none")
    else:
        R, t, inliers = ret
        t_err, R_err = metrics.relative_pose_error(T1to2, R, t, ignore_gt_t_thr=0.00001)
        R_errs.append(R_err)
        t_errs.append(t_err)
        print(t_err, R_err)



TxtPath="/root/data1/Loftr_base/LoFTR/data/megadepth/index/trainvaltest_list/val_list.txt" 

with open(TxtPath,'r') as file:
    txtfiles= [line.rstrip('\n') for line in file]

for txtname in txtfiles:

    root_dir="/root/data1/Loftr_base/LoFTR/data/megadepth/train"
    folder_path="/root/data1/Loftr_base/LoFTR/data/megadepth/index/scene_info_val_1500/"+txtname+".npz"


    scene_info = np.load(folder_path, allow_pickle=True)
    pair_infos=scene_info['pair_infos']

    i=0
    for pair_info in pair_infos:
        [idx0F, idx1F] = pair_info[0]  #这里改一下数据组织方式把，不知道原来的megadepth是怎么办到的。
        (idx0, idx1)=[int(idx0F),int(idx1F)]

        img_name0 = osp.join(root_dir, scene_info['image_paths'][idx0])
        img_name1 = osp.join(root_dir, scene_info['image_paths'][idx1])

        img0 = cv2.imread(img_name0,cv2.IMREAD_GRAYSCALE)#导入灰度图像  会不会图像很大匹配很好？
        img1 = cv2.imread(img_name1,cv2.IMREAD_GRAYSCALE)


        K_0 = np.array(scene_info['intrinsics'][idx0].copy(), dtype=np.float).reshape(3, 3)
        K_1 = np.array(scene_info['intrinsics'][idx1].copy(), dtype=np.float).reshape(3, 3)

        T0 = scene_info['poses'][idx0]
        T1 = scene_info['poses'][idx1]
        T_0to1 = np.array(np.matmul(T1, np.linalg.inv(T0)), dtype=np.float)[:4, :4]  # (4, 4)、

        #ORBPoseErro(img0,img1,K_0,K_1,T_0to1)
        SIFTPoseErro(img0,img1,K_0,K_1,T_0to1)
        i+=1
        print(txtname ,"total: ",len(pair_infos), "cur: ",i)

angular_thresholds = [5, 10, 20]
pose_errors = np.max(np.stack([R_errs, t_errs]), axis=0) #['identifiers'] 重复筛选没做，有问题在查
# pose_errors = np.min(np.stack([R_errs, t_errs]), axis=0) 
aucs = metrics.error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

print(aucs)

