import numpy as np
import json
from functools import lru_cache
import cv2
import pickle
from tqdm import tqdm
from typing import List, Dict

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler


from kp_preprocess import get_2d3d_perspective_transform, get_points_bbox, get_points_center_scale


class HandMeshEvalDataset(Dataset):
    def __init__(self, json_path, img_size=(224, 224), scale_enlarge=1.2, rot_angle=0):
        super().__init__()

        with open(json_path) as f:
            self.all_image_info = json.load(f)
        self.all_info = [{"image_path": image_path} for image_path in self.all_image_info]
        self.img_size = img_size
        self.scale_enlarge = scale_enlarge
        self.rot_angle = rot_angle

    def __len__(self):
        return len(self.all_image_info)
    
    def read_image(self, img_path):
        img = cv2.imread(img_path)
        return img

    def read_info(self, img_path):
        info_path = img_path.replace('.jpg', '.json')
        with open(info_path) as f:
            info = json.load(f)
        return info
    
    def __getitem__(self, index):
        image_path = self.all_image_info[index]
        img = self.read_image(image_path)
        data_dict = self.read_info(image_path)
        h, w = img.shape[:2]
        K = np.array(data_dict['K'])
        if "uv" in data_dict:
            uv = np.array(data_dict['uv'])
            xyz = np.array(data_dict['xyz'])
            vertices = np.array(data_dict['vertices'])
            uv_norm = uv.copy()
            uv_norm[:, 0] /= w   
            uv_norm[:, 1] /= h

            coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32") # Nx2x21x2
            coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

            valid_points = [uv[i] for i in range(len(uv)) if coord_valid[i]==1]        
            if len(valid_points) <= 1:
                valid_points = uv

            points = np.array(valid_points)
            min_coord = points.min(axis=0)
            max_coord = points.max(axis=0)
            center = (max_coord + min_coord)/2
            scale = max_coord - min_coord
        else:
            bbox = data_dict['bbox']
            x1, y1, x2, y2 = bbox[:4]
            center = np.array([(x1 + x2)/2, (y1 + y2) / 2])
            scale = np.array([x2 - x1, y2- y1])
            uv = np.zeros((21, 2), dtype=np.float32)
            xyz = np.zeros((21, 3), dtype=np.float32)
        
        ori_xyz = xyz.copy()
        ori_vertices = vertices.copy()
        scale = scale * self.scale_enlarge
        # perspective trans
        new_K, trans_matrix_2d, trans_matrix_3d = get_2d3d_perspective_transform(K, center, scale, self.rot_angle, self.img_size[0])
        img_processed = cv2.warpPerspective(img, trans_matrix_2d, self.img_size)
        new_uv = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        new_uv = (trans_matrix_2d @ new_uv.T).T
        new_uv = new_uv[:, :2] / new_uv[:, 2:]
        new_xyz = (trans_matrix_3d @ xyz.T).T       
        
        vertices = trans_matrix_3d.dot(vertices.T).T

         

        if img_processed.ndim == 2:
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
        img_processed = np.transpose(img_processed, (2, 0, 1))
        return {
            "img": np.ascontiguousarray(img_processed),
            "trans_matrix_2d": trans_matrix_2d,
            "trans_matrix_3d": trans_matrix_3d,            
            "K": new_K,
            "uv": new_uv,
            "xyz": new_xyz,
            "vertices": vertices,            
            "scale": self.img_size[0],
            "ori_xyz":ori_xyz,
            "ori_vertices":ori_vertices,

        }
        
    def __str__(self):
        return json.dumps(len(self.all_image_info))
