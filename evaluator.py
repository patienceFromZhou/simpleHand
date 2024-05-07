import torch
# import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from eval_datataset import HandMeshEvalDataset
from torch.utils.data import DataLoader, SequentialSampler

from tqdm import tqdm 
import refile
import os

from cfg import _CONFIG
from hand_net import HandNet
from utils import get_log_model_dir
from eval_utils import keypoint_auc, warp_uv_inverse, keypoint_mpjpe, vertice_pve
from models.losses import mesh_to_joints

class Evaluator:

    def __init__(self, dataloader, img_size=(224, 224), val_root_index=0) -> None:
        
        # self.is_main_process = dist.get_rank() == 0
        # self.local_rank = dist.get_rank()
        self.dataloader = dataloader
        self.img_size = img_size[0]
        
        self.root_index = val_root_index
        self.train_root_index = _CONFIG['DATA'].get('ROOT_INDEX', 9)

    def evaluate(self, model, distributed=False, is_tm=False):
        model.cuda()
        model.eval()
        data_list = []

        # progress_bar = tqdm if self.is_main_process else iter
        progress_bar = tqdm

        mask = np.ones((1, 21), dtype=bool)

        for cur_iter, batch_data in enumerate(progress_bar(self.dataloader)):
            for k in batch_data:
                batch_data[k] = batch_data[k].cuda().float()

            image = batch_data['img']
            joints_gt = batch_data['xyz']
            gt_uv = batch_data['uv']
            scale = batch_data['scale']
            K = batch_data['K']
            trans_matrix_2d = batch_data['trans_matrix_2d']
            vertices_gt = batch_data['vertices']

            root_depth_gt = joints_gt[:, self.train_root_index, 2]

            with torch.no_grad():
                res = model(image)
                joints = res["joints"]
                uv = res["uv"]
                # gamma = res["root_depth"]
                vertices = res['vertices']
                joints_mesh = mesh_to_joints(vertices)
                

            joints = joints_mesh.reshape(-1, 21, 3)
            uv = uv.reshape(-1, 21, 2) * self.img_size

            cur_res_list = []
            
            joints_gt = joints_gt - joints_gt[:, self.root_index:self.root_index+1]
            joints = joints - joints[:, self.root_index:self.root_index+1]

            trans_matrix_2d = trans_matrix_2d.cpu().numpy()
            pred_uv = uv.cpu().numpy()
            gt_uv = gt_uv.cpu().numpy()
            joints_gt = joints_gt.cpu().numpy()
            vertices_gt = vertices_gt.cpu().numpy()

            joints = joints.cpu().numpy()
            vertices = vertices.cpu().numpy()
            
            for i in range(joints_gt.shape[0]):
                cur_pred_uv = warp_uv_inverse(pred_uv[i], trans_matrix_2d[i])
                cur_gt_uv = warp_uv_inverse(gt_uv[i], trans_matrix_2d[i])
                cur_auc = keypoint_auc(cur_pred_uv[None], cur_gt_uv[None], mask)

                pa_mpjpe = keypoint_mpjpe(joints[i][None], joints_gt[i][None], mask, alignment="procrustes")
                mpjpe = keypoint_mpjpe(joints[i][None], joints_gt[i][None], mask)
                pa_mpvpe = vertice_pve(vertices[i][None], vertices_gt[i][None], alignment="procrustes")

                # cur_res_list.append([cur_auc, depth_dist[i], pa_mpjpe, mpjpe, pa_mpvpe])
                cur_res_list.append([cur_auc, pa_mpjpe, mpjpe, pa_mpvpe])
                


            data_list.extend(cur_res_list)

        # if distributed:
        #     results = gather_pyobj(data_list, obj_name="data_list", target_rank_id=0)

        #     if self.is_main_process:
        #         for x in results[1:]:
        #             data_list.extend(x)
        
        all_uv_auc = [data[0] for data in data_list]
        # all_depth_dist = [data[1] for data in data_list]
        all_pa_mpjpe_dist = [data[1] for data in data_list]
        all_mpjpe_dist = [data[2] for data in data_list]
        all_pa_mpvpe_dist = [data[3] for data in data_list]
        

        # dist.group_barrier()
        
        # if not self.is_main_process:
        #     return {}

        result_dict = {
            'auc': np.mean(all_uv_auc),
            # 'depth_dist': np.mean(all_depth_dist),
            'pa_mpjpe':np.mean(all_pa_mpjpe_dist),
            'mpjpe':np.mean(all_mpjpe_dist),
            "pa_mpvpe": np.mean(all_pa_mpvpe_dist) 
        }
        return result_dict


def build_evaluator(val_cfg):
    bmks = val_cfg["BMKS"]

    evaluator_dict = {}
    for bmk in bmks:
        dataset = HandMeshEvalDataset(bmk['json_path'], val_cfg["IMAGE_SHAPE"], bmk["scale_enlarge"])
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=val_cfg["BATCH_SIZE"], num_workers=2, timeout=120)

        evaluator = Evaluator(dataloader, img_size=val_cfg["IMAGE_SHAPE"], val_root_index=val_cfg.get("ROOT_INDEX", 9))
        evaluator_dict[bmk['name']] = evaluator

    return evaluator_dict


def infer(epoch):
    log_model_dir = get_log_model_dir(_CONFIG['NAME'])
    model_path = os.path.join(log_model_dir, epoch)
    print(model_path)
    model = HandNet(_CONFIG, pretrained=False)

    checkpoint = torch.load(refile.smart_open(model_path, "rb"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    evaluator_dict = build_evaluator(_CONFIG["VAL"])

    for bmk_name in evaluator_dict:
        evaluator = evaluator_dict[bmk_name]
        result_dict = evaluator.evaluate(model, True)
        print(bmk_name)
        print(result_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--infer_all", action="store_true")
    parser.add_argument("--epoch", type=str, default="latest")

    args = parser.parse_args()
    infer(args.epoch)
