import numpy as np
import cv2


def get_points_center_scale(points):
    # points must be NxC
    points = np.array(points)
    min_coord = points.min(axis=0)
    max_coord = points.max(axis=0)
    center = (max_coord + min_coord)/2
    scale = (max_coord - min_coord).max()
    return center, scale

def get_points_bbox(points):
    points = np.array(points, dtype="float32")
    min_coord = points.min(axis=0)
    max_coord = points.max(axis=0)
    return [*min_coord, *max_coord]
    

def preprocess(img, bbox, img_size=(128, 128), scale_enlarge=1.2):
    res_x, res_y = img_size

    # get bbox center and scale
    center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
    scale = max(bbox[2]-bbox[0], bbox[3]-bbox[1])
    scale = scale_enlarge * scale

    # generate warp M
    M = np.zeros((3, 3), dtype="float32")
    M[0, 0] = float(res_x) / scale
    M[1, 1] = float(res_y) / scale
    M[0, 2] = res_x * (-float(center[0]) / scale + .5)
    M[1, 2] = res_y * (-float(center[1]) / scale + .5)
    M[2, 2] = 1

    # warp img
    warp_img = cv2.warpPerspective(img, M, dsize=(res_x, res_y))

    return warp_img, M, scale


def projectPoints(xyz, K):
    """Project 3D coordinates into image space.
    Args:
        xyz: 3d点，shape=Nx3
        K: 相机参数，shape=3x3

    Returns:
        uv: 投影得到的2d点 shape=Nx2
    """
    xyz = np.array(xyz)[:, :3]
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def cross(v1, v2):
    x1, y1, z1 = v1
    x2, y2, z2 = v2

    a = y1 * z2 - y2 * z1
    b = z1 * x2 - z2 * x1
    c = x1 * y2 - x2 * y1

    cross_vector = np.array([a, b, c], dtype="float32")  
    return cross_vector

def cal_rot_mat_by_vector(vec1, vec2):
    norm_vec1 = vec1 / (np.linalg.norm(vec1) + 1e-7)
    norm_vec2 = vec2 / (np.linalg.norm(vec2) + 1e-7)

    cross_vec = cross(norm_vec1, norm_vec2)
    sin = np.linalg.norm(cross_vec)
    cos = (norm_vec1 * norm_vec2).sum()

    v1, v2, v3 = cross_vec

    V = np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ], dtype="float32")

    R = np.eye(3, dtype="float32") + V + V.dot(V) * (1 - cos) / sin**2
    return R


def get_trans_mat_by_center_K(center, K):
    fx = K[0][0]
    fy = K[1][1]
    dx = K[0][2]
    dy = K[1][2]

    vec1 = np.array([0, 0, 1], dtype="float32") # 原始相机方向
    vec2 = np.array([(center[0] - dx)/ fx, (center[1] - dy) / fy, 1], dtype="float32") # 变换后相机方向

    rot_mat = cal_rot_mat_by_vector(vec2, vec1)
    return rot_mat


def trans_3d_by_center_K(center, K, xyz):
    rot_mat = get_trans_mat_by_center_K(center, K)
    new_xyz = rot_mat.dot(xyz.T).T
    return new_xyz


def cal_perspective_mat(center, scale, res, K, new_K=None):
    u1 = center[0] - scale[0] / 2
    v1 = center[1] - scale[1] / 2
    u2 = center[0] + scale[0] / 2
    v2 = center[1] + scale[1] / 2

    fx = K[0][0]
    fy = K[1][1]
    dx = K[0][2]
    dy = K[1][2]

    x1 = (u1 - dx) / fx
    x2 = (u2 - dx) / fx
    y1 = (v1 - dy) / fy
    y2 = (v2 - dy) / fy

    rot_mat = get_trans_mat_by_center_K(center, K)

    xyz_array = np.array([
        [x1, y1, 1],
        [x1, y2, 1],
        [x2, y1, 1],
        [x2, y2, 1],
    ], dtype="float32")

    rot_xyz = (rot_mat@xyz_array.T).T

    if new_K is None:
        # 如果没有提供K，这里直接先预先投影一次，得到投影后的点，然后计算其外接框，得到scale

        temp_uv = projectPoints(rot_xyz, K)
        scale = (temp_uv.max(axis=0) - temp_uv.min(axis=0)).max()
        focal_scale = res / scale 
        fx = K[0][0]
        fy = K[1][1]

        new_K = np.array([
            [fx * focal_scale, 0, res/2],
            [0, fy * focal_scale, res/2],
            [0, 0, 1],
        ], dtype='float32')    

    proj_uv = projectPoints(rot_xyz, new_K)

    ori_uv = np.array([
        [u1, v1],
        [u1, v2],
        [u2, v1],
        [u2, v2],
    ], dtype="float32")

    # 根据两个图的点计算透视变换矩阵

    M = cv2.getPerspectiveTransform(ori_uv, proj_uv)
    return M, new_K


def trans2d_perspective(uv, M):
    pad = np.ones((uv.shape[0], 1))
    new_uv = np.concatenate([uv, pad], axis=1)
    trans_coord = M.dot(new_uv.T).T
    # 透视变换还需要除以第三项
    trans_coord = trans_coord / trans_coord[:, 2:3]
    return trans_coord[:, :2]


def get_2d3d_perspective_transform(K, center, scale, rot=0, res=224):
    """根据相机参数以及crop的中心点, crop大小, 以及旋转角度, 得到新相机视角下新相机参数K, 以及对应的2d, 3d透视变换矩阵

    Args:
        K (np.array): 3x3相机参数矩阵
        center (list or tuple):(x, y) crop 中心 
        scale (float): 抠图大小 
        rot (int, optional): 旋转角度. Defaults to 0.
        res (int, optional): 输出图大小. Defaults to 224.

    Returns:
        _type_: _description_
    """
    
    if isinstance(scale, int) or isinstance(scale, float):
        scale = [scale, scale]

    M, new_K = cal_perspective_mat(center, scale, res, K, None)

    camera_rot_mat = get_trans_mat_by_center_K(center, K)

    # rotate and rescale
    rot_mat = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot), np.cos(rot), 0],
        [0, 0, 1],
    ], dtype="float32")

    mat1 = np.array([
        [1, 0, -new_K[0][2]],
        [0, 1, -new_K[1][2]],
        [0, 0, 1],
    ], dtype="float32") # offset

    mat2 = np.array([
        [1, 0, new_K[0][2]],
        [0, 1, new_K[1][2]],
        [0, 0, 1],
    ], dtype="float32") # offset

    rot_2d = mat2 @ rot_mat @ mat1
    final_M_2d = rot_2d @ M
    final_M_3d = rot_mat @ camera_rot_mat


    return new_K, final_M_2d, final_M_3d