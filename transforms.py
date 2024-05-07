# https://github.com/open-mmlab/mmhuman3d/blob/1dd2d281a775a2da197074ace698be324f3f8196/mmhuman3d/data/datasets/pipelines/transforms.py#L286

import math
import cv2
import numpy as np


def get_points_center_scale(points):
    # points must be NxC
    points = np.array(points)
    min_coord = points.min(axis=0)
    max_coord = points.max(axis=0)
    center = (max_coord + min_coord)/2
    scale = (max_coord - min_coord).max()
    return center, scale

def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.
    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian
    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.
    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.
    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)
    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt

def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.
    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform
    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

    return new_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False,
                         pixel_std=1.0):
    """Get the affine transform matrix, given the center/scale/rot/output_size.
    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    scale_tmp = scale * pixel_std

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).
    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].
    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return 



def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.
    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.
    Returns:
        matrix (np.ndarray[..., 2]): Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)


def _flip_keypoints(keypoints, img_width=None):
    """Flip human joints horizontally.
    Note:
        num_keypoints: K
        num_dimension: D
    Args:
        keypoints (np.ndarray([K, D])): Coordinates of keypoints.
        img_width (int | None, optional): The width of the original image.
            To flip 2D keypoints, image width is needed. To flip 3D keypoints,
            we simply negate the value of x-axis. Default: None.
    Returns:
        keypoints_flipped
    """

    keypoints_flipped = keypoints.copy()

    # Flip horizontally
    if img_width is None:
        keypoints_flipped[:, 0] = -keypoints_flipped[:, 0]
    else:
        keypoints_flipped[:, 0] = img_width - 1 - keypoints_flipped[:, 0]

    return keypoints_flipped

def _flip_hand_pose(pose):
    dim_flip = np.array([1, -1, -1], dtype=pose.dtype)
    pose = pose* dim_flip
    return pose

def _flip_axis_angle(r):
    """Flip axis_angle horizontally.
    Args:
        r (np.ndarray([3]))
    Returns:
        f_flipped
    """
    dim_flip = np.array([1, -1, -1], dtype=r.dtype)
    r = r * dim_flip
    return r

def _construct_rotation_matrix(rot, size=3):
    """Construct the in-plane rotation matrix.
    Args:
        rot (float): Rotation angle (degree).
        size (int): The size of the rotation matrix.
            Candidate Values: 2, 3. Defaults to 3.
    Returns:
        rot_mat (np.ndarray([size, size]): Rotation matrix.
    """
    rot_mat = np.eye(size, dtype=np.float32)
    if rot != 0:
        rot_rad = np.deg2rad(rot)
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]

    return rot_mat

def _rotate_joints_3d(joints_3d, rot):
    """Rotate the 3D joints in the local coordinates.
    Notes:
        Joints number: K
    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        rot (float): Rotation angle (degree).
    Returns:
        joints_3d_rotated
    """
    # in-plane rotation
    # 3D joints are rotated counterclockwise,
    # so the rot angle is inversed.
    rot_mat = _construct_rotation_matrix(-rot, 3)

    joints_3d_rotated = np.einsum('ij,kj->ki', rot_mat, joints_3d)
    joints_3d_rotated = joints_3d_rotated.astype('float32')
    return joints_3d_rotated

def _rotate_smpl_pose(pose, rot):
    """Rotate SMPL pose parameters.
    SMPL (https://smpl.is.tue.mpg.de/) is a 3D
    human model.
    Args:
        pose (np.ndarray([72])): SMPL pose parameters
        rot (float): Rotation angle (degree).
    Returns:
        pose_rotated
    """
    pose_rotated = pose.copy()
    if rot != 0:
        rot_mat = _construct_rotation_matrix(-rot)
        orient = pose[:3]
        # find the rotation of the body in camera frame
        per_rdg, _ = cv2.Rodrigues(orient.astype(np.float32))
        # apply the global rotation to the global orientation
        res_rot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
        pose_rotated[:3] = (res_rot.T)[0]

    return pose_rotated

class RandomHorizontalFlip(object):
    """Flip the image randomly.
    Flip the image randomly based on flip probaility.
    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
    """

    def __init__(self, flip_prob=0.5):
        assert 0 <= flip_prob <= 1
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Call function to flip image and annotations.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip' key is added into
                result dict.
        """
        if np.random.rand() > self.flip_prob:
            results['is_flipped'] = np.array([0])
            return results

        results['is_flipped'] = np.array([1])

        # flip image
        for key in results.get('img_fields', ['img']):
            # results[key] = mmcv.imflip(results[key], direction='horizontal')
            results[key] = cv2.flip(results[key], 1)

        # flip keypoints2d
        if 'keypoints2d' in results:
            # assert self.flip_pairs is not None
            width = results['img'][:, ::-1, :].shape[1]
            keypoints2d = results['keypoints2d'].copy()
            keypoints2d = _flip_keypoints(keypoints2d, width)
            results['keypoints2d'] = keypoints2d

        # flip bbox center
        center = results['center']
        center[0] = width - 1 - center[0]
        results['center'] = center

        # flip keypoints3d
        if 'keypoints3d' in results:
            # assert self.flip_pairs is not None
            keypoints3d = results['keypoints3d'].copy()
            keypoints3d = _flip_keypoints(keypoints3d)
            results['keypoints3d'] = keypoints3d

        if "vertices" in results:
            vertices = results['vertices'].copy()
            vertices = _flip_keypoints(vertices)
            results['vertices'] = vertices
        
        # todo: support two hand flip
        if "mano_pose" in results:
            mano_pose = results['mano_pose'].copy()            
            mano_pose = _flip_hand_pose(mano_pose.reshape(-1, 3)).reshape(-1)
            
            results['mano_pose'] = mano_pose
            
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'
    

class GetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.
    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.
    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=30, min_scale_factor=0.9, max_scale_factor=1.3, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        rf = self.rot_factor

        s_factor = np.random.rand() * (self.max_scale_factor -
                                        self.min_scale_factor) + self.min_scale_factor        
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0.0

        results['scale'] = s
        results['rotation'] = r

        return results


class MeshAffine:
    """Affine transform the image to get input image.
    Affine transform the 2D keypoints, 3D kepoints. Required keys: 'img',
    'pose', 'img_shape', 'rotation' and 'center'. Modifies key: 'img',
    ''keypoints2d', 'keypoints3d', 'pose'.
    """

    def __init__(self, img_res):
        self.img_res = img_res
        self.image_size = np.array([img_res, img_res])

    def __call__(self, results):
        c = results['center']
        s = results['scale']
        r = results['rotation']
        trans = get_affine_transform(c, s, r, self.image_size)

        if 'img' in results:
            img = results['img']

            # img before affine
            # ori_img = img.copy()
            # results['crop_transform'] = trans
            # results['ori_img'] = ori_img
            # results['img_fields'] = ['img', 'ori_img']

            img = cv2.warpAffine(
                img,
                trans, (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)
            results['img'] = img

        if 'keypoints2d' in results:
            keypoints2d = results['keypoints2d'].copy()
            num_keypoints = len(keypoints2d)
            for i in range(num_keypoints):
                # if keypoints2d[i][2] > 0.0:
                #     keypoints2d[i][:2] = \
                #         affine_transform(keypoints2d[i][:2], trans)
                keypoints2d[i][:2] = \
                    affine_transform(keypoints2d[i][:2], trans)                    
            results['keypoints2d'] = keypoints2d

        if 'keypoints3d' in results:
            keypoints3d = results['keypoints3d'].copy()
            keypoints3d[:, :3] = _rotate_joints_3d(keypoints3d[:, :3], r)
            results['keypoints3d'] = keypoints3d

        if "vertices" in results:
            vertices = results['vertices'].copy()
            vertices[:, :3] = _rotate_joints_3d(vertices[:, :3], r)
            results['vertices'] = vertices
            
        if "mano_pose" in results:
            mano_pose = results['mano_pose'].copy()
            mano_pose = _rotate_smpl_pose(mano_pose, r)
            results['mano_pose'] = mano_pose            

        return results
    
class BBoxCenterJitter(object):

    def __init__(self, factor=0.0, dist='normal'):
        super(BBoxCenterJitter, self).__init__()
        self.factor = factor
        self.dist = dist
        assert self.dist in [
            'normal', 'uniform'
        ], (f'Distribution must be normal or uniform, not {self.dist}')

    def __call__(self, results):
        # body model: no process
        if self.factor <= 1e-3:
            return results

        bbox_size = results['scale'][0]

        jitter = bbox_size * self.factor

        if self.dist == 'normal':
            center_jitter = np.random.randn(2) * jitter
        elif self.dist == 'uniform':
            center_jitter = np.random.rand(2) * 2 * jitter - jitter

        center = results['center']
        H, W = results['img'].shape[:2]
        new_center = center + center_jitter
        new_center[0] = np.clip(new_center[0], 0, W)
        new_center[1] = np.clip(new_center[1], 0, H)

        results['center'] = new_center
        return results    
    

class RandomChannelNoise:
    """Data augmentation with random channel noise.
    Required keys: 'img'
    Modifies key: 'img'
    Args:
        noise_factor (float): Multiply each channel with
         a factor between``[1-scale_factor, 1+scale_factor]``
    """

    def __init__(self, noise_factor=0.4):
        self.noise_factor = noise_factor

    def __call__(self, results):
        """Perform data augmentation with random channel noise."""
        img = results['img']

        # Each channel is multiplied with a number
        # in the area [1-self.noise_factor, 1+self.noise_factor]
        pn = np.random.uniform(1 - self.noise_factor, 1 + self.noise_factor,
                               (1, 3))
        img = cv2.multiply(img, pn)

        results['img'] = img

        if 'ori_img' in results:
            img = results['ori_img']
            img = cv2.multiply(img, pn)

            results['ori_img'] = img

        return results
    

class SimulateLowRes(object):

    def __init__(self,
                 dist: str = 'categorical',
                 factor: float = 1.0,
                 cat_factors=(1.0, ),
                 factor_min: float = 1.0,
                 factor_max: float = 1.0) -> None:
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.dist = dist
        self.cat_factors = cat_factors
        assert dist in ['uniform', 'categorical']

    def _sample_low_res(self, image: np.ndarray) -> np.ndarray:
        """"""
        if self.dist == 'uniform':
            downsample = self.factor_min != self.factor_max
            if not downsample:
                return image
            factor = np.random.rand() * (self.factor_max -
                                         self.factor_min) + self.factor_min
        elif self.dist == 'categorical':
            if len(self.cat_factors) < 2:
                return image
            idx = np.random.randint(0, len(self.cat_factors))
            factor = self.cat_factors[idx]

        H, W, _ = image.shape
        downsampled_image = cv2.resize(image,
                                       (int(W // factor), int(H // factor)),
                                       cv2.INTER_NEAREST)
        resized_image = cv2.resize(downsampled_image, (W, H),
                                   cv2.INTER_LINEAR_EXACT)
        return resized_image

    def __call__(self, results):
        """"""
        img = results['img']
        img = self._sample_low_res(img)
        results['img'] = img

        return results


from kp_preprocess import get_2d3d_perspective_transform

def trans2d_perspective(uv, M):
    pad = np.ones((uv.shape[0], 1))
    new_uv = np.concatenate([uv, pad], axis=1)
    trans_coord = M.dot(new_uv.T).T
    # 透视变换还需要除以第三项
    trans_coord = trans_coord / trans_coord[:, 2:3]
    return trans_coord[:, :2]


class MeshPerspectiveTransform(object):

    def __init__(self, img_res):
        self.img_res = img_res
        self.image_size = np.array([img_res, img_res])


    def __call__(self, results):
        c = results['center']
        s = results['scale']
        r = results['rotation']
        K = results['K']

        img = results['img']

        new_K, trans_matrix_2d, trans_matrix_3d = get_2d3d_perspective_transform(K, c, s, r, self.image_size[0])
        warp_img = cv2.warpPerspective(img, trans_matrix_2d, (self.image_size[0], self.image_size[1]))

        uv = results['keypoints2d']
        xyz = results['keypoints3d']
        new_uv = trans2d_perspective(uv, trans_matrix_2d)
        new_xyz = trans_matrix_3d.dot(xyz.T).T

        results['keypoints2d'] = new_uv
        results['keypoints3d'] = new_xyz
        results['img'] = warp_img
        results["K"] = new_K

        if "vertices" in results:
            vertices = results['vertices'].copy()
            vertices = trans_matrix_3d.dot(vertices.T).T
            results['vertices'] = vertices
            
        if "mano_pose" in results:
            mano_pose = results['mano_pose'].copy()
                
            orient = mano_pose[:3]
            # find the rotation of the body in camera frame
            per_rdg, _ = cv2.Rodrigues(orient.astype(np.float32))
            # apply the global rotation to the global orientation
            res_rot, _ = cv2.Rodrigues(np.dot(trans_matrix_3d, per_rdg))
            mano_pose[:3] = (res_rot.T)[0]
            
            results['mano_pose'] = mano_pose            
            

        return results
