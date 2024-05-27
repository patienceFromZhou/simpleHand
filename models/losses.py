from functools import lru_cache
import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch import Tensor

from .mano_torch import vertices2joints


NUM_BODY_JOINTS = 1
NUM_HAND_JOINTS = 15
NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS
NUM_SHAPES = 10
USE_LEFT = False
MANO_PARAMS_PATH = "models/"

def load_mano_params_c(use_left=True):
    if use_left:
        mano_path = os.path.join(MANO_PARAMS_PATH, "MANO_LEFT_C.pkl")
    else:
        mano_path = os.path.join(MANO_PARAMS_PATH, "MANO_RIGHT_C.pkl")
    
    with open(mano_path, 'rb') as mano_file:
        model_data = pickle.load(mano_file)
    return model_data

MANO_DATA_LEFT = load_mano_params_c(True)
MANO_DATA_RIGHT = load_mano_params_c(False)


@lru_cache(1)
def get_faces():
    faces = np.array(MANO_DATA_RIGHT["f"]).astype("int32")    
    return torch.from_numpy(faces)

def project_k(coord, K):
    b, j, c = coord.shape
    if c == 2:
        pad = torch.ones((b, j, 1)).to(coord.device)
        coord = torch.concat([coord, pad], dim=2)
    coord_t = torch.permute(coord, (0, 2, 1)) # NxJx3 -> Nx3xJ
    coord_t = torch.matmul(K, coord_t)
    coord_t = torch.permute(coord_t, (0, 2, 1))
    if c == 2:
        coord_t = coord_t[:, :, :2]
    return coord_t

def get_fingertip(vertices):
    '''
    原始MANO模型的joint不包含5个指尖, 这里从vertics得到五个指尖的位置
    '''
    idxs = [744, 320, 443, 555, 672]
    fingertips = vertices[:, idxs]

    return fingertips


def remap_joints_and_fingertip(joints, fingertips):
    '''
    将原始的MANO出的16个joints+5个fingertips进行重排
    '''
    new_idxs = [0, 
        13, 14, 15, 16, 
        1, 2, 3, 17,
        4, 5, 6, 18,
        10, 11, 12, 19,
        7, 8, 9, 20,
    ]
    new_joints = torch.concat([joints, fingertips], dim=1) # Bx21x3
    new_joints = new_joints[:, new_idxs]
    return new_joints    


def mesh_to_joints(mesh: torch.Tensor):
    J_reg = torch.tensor(np.array(MANO_DATA_RIGHT["J_regressor"]), dtype=torch.float32).to(mesh.device)
    joints = vertices2joints(J_reg, mesh)
    fingertips = get_fingertip(mesh)
    joints = remap_joints_and_fingertip(joints, fingertips)
    return joints        



def cal_joint_loss(pred, label, mask):
    joint_loss = (torch.abs(pred - label)).sum(dim=2).mean(dim=1)
    joint_loss = (joint_loss * mask).mean()
    bone_loss = cal_bone_len(pred, label, mask)
    return joint_loss, bone_loss    


def cal_bone_len(pred, label, mask):
    bone_index = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    start_bone_index = [i[0] for i in bone_index]
    end_bone_index = [i[1] for i in bone_index]

    start_joint_pred = pred[:, start_bone_index]
    end_joint_pred = pred[:, end_bone_index]

    start_joint_label = label[:, start_bone_index]
    end_joint_label = label[:, end_bone_index]

    pred_bone_len = ((start_joint_pred - end_joint_pred)**2).sum(dim=2)
    gt_bone_len = ((start_joint_label - end_joint_label)**2).sum(dim=2)

    bone_loss = ((pred_bone_len - gt_bone_len)**2).mean(dim=1)
    bone_loss = (mask * bone_loss).mean()
    return bone_loss    


def batch_cross(v1, v2):
    # NxCx3
    x1 = v1[:, :, 0:1]
    y1 = v1[:, :, 1:2]
    z1 = v1[:, :, 2:]
    x2 = v2[:, :, 0:1]
    y2 = v2[:, :, 1:2]
    z2 = v2[:, :, 2:]

    a = y1 * z2 - y2 * z1
    b = z1 * x2 - z2 * x1
    c = x1 * y2 - x2 * y1

    cross_vector = torch.concat([a, b, c], dim=2)
    return cross_vector


def gen_faces_point(vertices, faces):
    idx_p1 = faces[:, 0]
    idx_p2 = faces[:, 1]
    idx_p3 = faces[:, 2]

    p1 = vertices[:, idx_p1]
    p2 = vertices[:, idx_p2]
    p3 = vertices[:, idx_p3]
    return p1, p2, p3


def norm_vector(vec):
    len_s = len(list(vec.shape))
    v_len = torch.sqrt((vec*vec).sum(dim=len_s-1, keepdims=True) + 1e-9)
    vec = vec / v_len
    return vec


def gen_cross_product(vertices, faces, norm=False):
    p1, p2, p3 = gen_faces_point(vertices, faces)
    vp1p2 = p2 - p1
    vp1p3 = p3 - p1
    cross_v = batch_cross(vp1p2, vp1p3)

    if norm:
        cross_v_len = torch.sqrt((cross_v*cross_v).sum(dim=2, keepdims=True) + 1e-9)
        cross_v = cross_v / cross_v_len
    return cross_v


def norm_loss(pred_vertices, gt_vertices, faces):
    # 这个loss是为了式一个face上的三条vector垂直于其法向量
    nv_gt = gen_cross_product(gt_vertices, faces, True)
    nv_pred = gen_cross_product(pred_vertices, faces, True)

    v_loss = ((nv_pred - nv_gt)**2).sum(dim=2).mean(dim=1).mean()
    return v_loss


def edge_loss(pred_vertices:Tensor, gt_vertices: Tensor, faces: Tensor):
    """_summary_

    Args:
        pred_vertices (Tensor): [N, 778, 3]
        gt_vertices (Tensor): [N, 778, 3]
        faces (Tensor): [1536, 3]

    Returns:
        _type_: edge loss
    """
    p1, p2, p3 = gen_faces_point(pred_vertices, faces)
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p1 - p3

    p1_gt, p2_gt, p3_gt = gen_faces_point(gt_vertices, faces) # N, 1536, 3
    v1_gt = p2_gt - p1_gt
    v2_gt = p3_gt - p2_gt
    v3_gt = p1_gt - p3_gt

    v1_s = len(list(v1.shape))
    v1_len = torch.sqrt((v1*v1).sum(dim=v1_s-1, keepdims=True) + 1e-9) # N, 1536, 1
    v2_len = torch.sqrt((v2*v2).sum(dim=v1_s-1, keepdims=True) + 1e-9)
    v3_len = torch.sqrt((v3*v3).sum(dim=v1_s-1, keepdims=True) + 1e-9)
    v1_len_gt = torch.sqrt((v1_gt*v1_gt).sum(dim=v1_s-1, keepdims=True) + 1e-9)
    v2_len_gt = torch.sqrt((v2_gt*v2_gt).sum(dim=v1_s-1, keepdims=True) + 1e-9)
    v3_len_gt = torch.sqrt((v3_gt*v3_gt).sum(dim=v1_s-1, keepdims=True) + 1e-9)

    v1_loss = torch.abs((v1_len - v1_len_gt).sum(dim=2)).mean(dim=1).mean()
    v2_loss = torch.abs((v2_len - v2_len_gt).sum(dim=2)).mean(dim=1).mean()
    v3_loss = torch.abs((v3_len - v3_len_gt).sum(dim=2)).mean(dim=1).mean()

    v_loss = v1_loss + v2_loss + v3_loss
    return v_loss


def l1_loss(pred, gt, weight=None, valid=None):
    """l1 loss

    Args:
        pred : [B, J, C]
        gt :  [B, J, C]
        weight (optional): [B, J]. Defaults to None.
        valid (optional):  [B]. Defaults to None.
    Returns:
        l1 loss: float
    """
    loss = (torch.abs(pred - gt)).sum(dim=2)
    if weight is not None:
        loss = loss * weight
    loss = loss.mean(-1)
    if valid is not None:
        loss = loss * valid
    return loss.mean()    


    
def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2


def cal_xyz_dist_by_pa(gt_xyz, pred_xyz):
    gt_xyz_ = np.array(gt_xyz, dtype="float32")
    pred_xyz_ = np.array(pred_xyz, dtype="float32")
    new_pred_xyz_ = rigid_align(pred_xyz_, gt_xyz_)
    dist_xyz = np.linalg.norm(gt_xyz_-new_pred_xyz_, axis=1).reshape(-1)
    return dist_xyz
