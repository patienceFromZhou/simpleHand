from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import Tuple, List
import numpy as np

import torch
import torch.nn.functional as F
import os
import pickle

from typing import NewType, Union, Optional
Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


USE_LEFT = True
MANO_PARAMS_PATH = "s3://zouminqiang/hand6dof/hand_keypoint/models/MANO/weight/"


def load_mano_params_c(use_left=True):
    if use_left:
        mano_path = os.path.join(MANO_PARAMS_PATH, "MANO_LEFT_C.pkl")
    else:
        mano_path = os.path.join(MANO_PARAMS_PATH, "MANO_RIGHT_C.pkl")
    
    with open(mano_path, 'rb') as mano_file:
        model_data = pickle.load(mano_file)
    return model_data


MANO_DATA = load_mano_params_c(USE_LEFT)


def vertices2landmarks(
    vertices: Tensor,
    faces: Tensor,
    lmk_faces_idx: Tensor,
    lmk_bary_coords: Tensor
) -> Tensor:
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
) -> Tuple[Tensor, Tensor]:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def vertices2joints(J_regressor: Tensor, vertices: Tensor) -> Tensor:
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas: Tensor, shape_disps: Tensor) -> Tensor:
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def prepare_mano_params(model_data):
    torch_mano = {
        "hands_components": torch.tensor(np.array(model_data["hands_components"])),  # (45, 45)
        "f": torch.tensor(np.array(model_data["f"]).astype("int32"), dtype=torch.int32),  # (1538, 3)
        "J_regressor": torch.tensor(np.array(model_data["J_regressor"])),  # (16, 778)
        "kintree_table": torch.tensor(np.array(model_data["kintree_table"])),  # (2,16)
        "J": torch.tensor(np.array(model_data["J"])),  # (16, 3)
        "hands_coeffs": torch.tensor(np.array(model_data["hands_coeffs"])), # (1554, 45)
        "weights": torch.tensor(np.array(model_data["weights"])),  # (778, 16)
        "posedirs": torch.tensor(np.array(model_data["posedirs"]).reshape(-1, 135).T),  # (778, 3, 135) -> (135, 778*3)
        "hands_mean": torch.tensor(np.array(model_data["hands_mean"])[None]),  # (1, 45)
        "v_template": torch.tensor(np.array(model_data["v_template"])[None]),  # (1, 778, 3)
        "shapedirs": torch.tensor(np.array(model_data["shapedirs"])),  # (778, 3, 10)
    }
    return torch_mano

def MANO(betas, global_orient, hand_pose, num_pca_comps=45, use_mean_hand=False):
    dtype=torch.float32
    torch_mano = prepare_mano_params(MANO_DATA)

    if global_orient is None:
        batch_size = 1
        global_orient = torch.eye(3, dtype="").view(
            1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
    else:
        batch_size = global_orient.shape[0]
    if hand_pose is None:
        hand_pose = torch.eye(3, dtype=dtype).view(
            1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
    if betas is None:
        betas = torch.zeros(
            [batch_size, 10], dtype=dtype)

    full_pose = torch.cat([global_orient, hand_pose], dim=1)
    vertices, joints = lbs(betas, full_pose, torch_mano["v_template"],
                            torch_mano["shapedirs"], torch_mano["posedirs"],
                            torch_mano["J_regressor"], torch_mano["kintree_table"][0],
                            torch_mano["weights"], pose2rot=True)

    return vertices, joints, torch_mano["f"]
