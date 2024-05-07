import json
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import albumentations as A
from typing import List, Dict
from itertools import cycle
from cfg import _CONFIG
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transforms import GetRandomScaleRotation, MeshAffine, RandomHorizontalFlip, \
            get_points_center_scale, RandomChannelNoise, BBoxCenterJitter, MeshPerspectiveTransform


DATA_CFG = _CONFIG["DATA"]
IMAGE_SHAPE: List = DATA_CFG["IMAGE_SHAPE"][:2]
NORMALIZE_3D_GT = DATA_CFG['NORMALIZE_3D_GT']
AUG_CFG: Dict = DATA_CFG["AUG"]
ROOT_INDEX = DATA_CFG['ROOT_INDEX']

def read_info(img_path):
    info_path = img_path.replace('.jpg', '.json')
    with open(info_path) as f:
        info = json.load(f)
    return info

with open(DATA_CFG['JSON_DIR']) as f:
    all_image_info = json.load(f)
all_info = []
for image_path in tqdm(all_image_info):
    info = read_info(image_path)
    info['image_path'] = image_path
    all_info.append(info)

class HandDataset(Dataset):
    def __init__(self, all_info):
        super().__init__()

        self.init_aug_funcs()
        self.all_info = all_info

    def __len__(self):
        return len(self.all_info)
    
    def init_aug_funcs(self):
        self.random_channel_noise = RandomChannelNoise(**AUG_CFG['RandomChannelNoise'])
        self.random_bright = A.RandomBrightnessContrast(**AUG_CFG["RandomBrightnessContrastMap"])            
        self.random_flip = RandomHorizontalFlip(**AUG_CFG["RandomHorizontalFlip"])
        self.bbox_center_jitter = BBoxCenterJitter(**AUG_CFG["BBoxCenterJitter"])
        self.get_random_scale_rotation = GetRandomScaleRotation(**AUG_CFG["GetRandomScaleRotation"])
        self.mesh_affine = MeshAffine(IMAGE_SHAPE[0])
        self.mesh_perspective_trans = MeshPerspectiveTransform(IMAGE_SHAPE[0])
        
        self.root_index = ROOT_INDEX

    def read_image(self, img_path):
        img = cv2.imread(img_path)
        return img
    
    def __getitem__(self, index):
        data_info = self.all_info[index]
        img = self.read_image(data_info['image_path'])
        # keypoints2d = np.array(data_info['uv'], dtype=np.float32)
        keypoints3d = np.array(data_info['xyz'], dtype=np.float32)
        K = np.array(data_info['K'], dtype=np.float32)
        
        proj_points = (K @ keypoints3d.T).T
        keypoints2d = proj_points[:, :2] / (proj_points[:, 2:] + 1e-7)
        
        vertices = np.array(data_info['vertices']).astype('float32')

        h, w = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            

        uv_norm = keypoints2d.copy()
        uv_norm[:, 0] /= w   
        uv_norm[:, 1] /= h

        coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32") # Nx2x21x2
        coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

        valid_points = [keypoints2d[i] for i in range(len(keypoints2d)) if coord_valid[i]==1]
        
        points = np.array(valid_points)
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        center = (max_coord + min_coord)/2
        scale = max_coord - min_coord
                
        results = {
            "img": img,
            "keypoints2d": keypoints2d,
            "keypoints3d": keypoints3d,
            "vertices": vertices,
            
            "center": center,
            "scale": scale,
            "K": K,
        }
        
        # 1. Crop and Rot
        results = self.bbox_center_jitter(results)
        results = self.get_random_scale_rotation(results)
        # results = self.mesh_affine(results)
        results = self.mesh_perspective_trans(results)

        # 2. 3D KP Root Relative
        root_point = results['keypoints3d'][self.root_index].copy()
        results['keypoints3d'] = results['keypoints3d'] - root_point[None, :]
        results['vertices'] = results['vertices'] - root_point[None, :]
        
        hand_img_len = IMAGE_SHAPE[0]
        root_depth = root_point[2]

        hand_world_len = 0.2
        fx = results['K'][0][0]
        fy = results['K'][1][1]
        camare_relative_k = np.sqrt(fx * fy * (hand_world_len**2) / (hand_img_len**2))
        gamma = root_depth / camare_relative_k
        # 3. Random Flip 
        results = self.random_flip(results)
        # 4. Image aug
        results = self.random_channel_noise(results)
        results['img'] = self.random_bright(image=results['img'])['image']

        trans_uv = results["keypoints2d"]
        trans_uv[:, 0] /= IMAGE_SHAPE[0]
        trans_uv[:, 1] /= IMAGE_SHAPE[1]

        trans_coord_valid = (trans_uv > 0).astype("float32") * (trans_uv < 1).astype("float32") # Nx2x21x2
        trans_coord_valid = trans_coord_valid[:, 0] * trans_coord_valid[:, 1]
        trans_coord_valid *= coord_valid

        xyz = results["keypoints3d"]
        if NORMALIZE_3D_GT:
            joints_bone_len = np.sqrt(((xyz[0:1] - xyz[9:10])**2).sum(axis=-1, keepdims=True) + 1e-8)
            xyz = xyz  / joints_bone_len
        
        xyz_valid = 1

        if trans_coord_valid[9] == 0 and trans_coord_valid[0] == 0:
            xyz_valid = 0

        img = results['img']
        img = np.transpose(img, (2,0,1))
        data = {
            "img": img,
            "uv": results["keypoints2d"],
            "xyz": xyz,
            "vertices": results['vertices'],                
            "uv_valid": trans_coord_valid,
            "gamma": gamma,
            "xyz_valid": xyz_valid,
        }

        return data

def build_train_loader(batch_size):
	dataset = HandDataset(all_info)
	sampler = RandomSampler(dataset, replacement=True)
	dataloader = (DataLoader(dataset, batch_size=batch_size, sampler=sampler))
	return iter(dataloader)

# if __name__ == "__main__":
#     train_loader = build_train_loader(_CONFIG['TRAIN']['DATALOADER']['MINIBATCH_SIZE_PER_DIVICE'])
#     batch = next(train_loader)
#     with open('batch_data.pkl', 'rb') as f:
#         pickle.dump(batch, f)
#     from IPython import embed 
#     embed()
#     exit()