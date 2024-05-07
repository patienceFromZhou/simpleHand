import argparse
import os
import subprocess
import time
from loguru import logger
import numpy as np

import torch
import torch.distributed as dist
import torch.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import random
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from hand_net import HandNet
from cfg import _CONFIG
from dataset import build_train_loader
from utils import GPUMemoryMonitor, get_log_model_dir

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def get_learning_rate(epoch, step, base_lr, minibatch_per_epoch, warmup_epoch, stop_epoch):
    final_lr = 0.0
    warmup_iter = minibatch_per_epoch * warmup_epoch
    warmup_lr_schedule = np.linspace(0, base_lr, warmup_iter)
    decay_iter = minibatch_per_epoch * (stop_epoch - warmup_epoch)
    if epoch < warmup_epoch:
        cur_lr = warmup_lr_schedule[step + epoch*minibatch_per_epoch]
    else:
        if epoch < stop_epoch // 2:
            return base_lr

        return base_lr / 10
        
    return cur_lr

class Trainer:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.log_model_dir = get_log_model_dir(cfg['NAME'])


        ngpus_per_node = torch.cuda.device_count()

        global_world_size = 1 * ngpus_per_node

        self.args.world_size = global_world_size
        self.args.gpu = int(os.environ['LOCAL_RANK'])
        self.args.local_rank = int(os.environ['LOCAL_RANK'])
        self.args.rank = self.args.local_rank
        
        self.rank = self.args.rank
        self.local_rank = self.args.local_rank

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def after_train(self):
        pass

    def before_train(self):
        logger.info(f"init group {self.rank}")
        if self.rank == 0:
            logger.add("./train_log/out.log", backtrace=True, diagnose=True)   
        self.set_random_seed(self.rank)
        self.max_epoch = self.cfg["TRAIN"]["MAX_EPOCH"]
        self.warmup_epoch =  self.cfg["TRAIN"]["WARMUP_EPOCH"]
        self.base_lr = self.cfg["TRAIN"]["BASE_LR"]
        self.weight_decay = self.cfg["TRAIN"]["WEIGHT_DECAY"]
        self.dump_epoch_interval = self.cfg["TRAIN"]["DUMP_EPOCH_INTERVAL"]

        args = self.args

        if args.local_rank != -1:
            args.distributed = True
            torch.cuda.set_device(args.local_rank)
            args.dist_backend = 'nccl'
            print('| distributed init world_size {}, (rank {}): , gpu {}'.format(
                args.world_size, args.rank, args.local_rank), flush=True)
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)            

            dist.barrier()

        if args.rank in [-1, 0]:
            self.writer = SummaryWriter(os.path.join("./train_log", 'train'))            

        model = HandNet(self.cfg)
        model.cuda(args.local_rank)

        decay = []
        no_decay = []
        bn_wd_skip = True
        for name, param in model.named_parameters():
            if ('bn' in name or 'bias' in name) and bn_wd_skip:
                no_decay.append(param)
            else:
                decay.append(param)

        per_param_args = [{'params': decay},
                        {'params': no_decay, 'weight_decay': 0.0}]

        self.optimizer = torch.optim.AdamW(per_param_args, lr=self.base_lr, weight_decay=self.weight_decay)
        self.grad_scaler = GradScaler()
        
        model = self.resume_train(model)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            broadcast_buffers=True, find_unused_parameters=True)

        self.train_loader = build_train_loader(self.cfg["TRAIN"]["DATALOADER"]["MINIBATCH_SIZE_PER_DIVICE"])
        self.max_iter = self.cfg["TRAIN"]["DATALOADER"]["MINIBATCH_PER_EPOCH"]
        
        self.gpu_monitor = GPUMemoryMonitor()
        self.model = model
        self.model.train()

        if self.rank == 0:
            logger.info("Training start...")
            logger.info("\n{}".format(model))


    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def profile(self):
        self.before_train()
        self.before_epoch()
        self.epoch = 0

        log_path = "./train_log/log"
        wait = 2
        warmup = 3
        active = 10
        repeat = 2
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        tb_handler = torch.profiler.tensorboard_trace_handler(log_path)
        with torch.profiler.profile(
            schedule=schedule, on_trace_ready=tb_handler,
            record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            for self.iter in range(self.max_iter):
                if self.iter >= (wait + warmup + active) * repeat:
                    break
                batch_data = next(self.train_loader)
                self.train_one_iter(batch_data)         
                prof.step() 


    def before_epoch(self):
        self.train_loader = build_train_loader(self.cfg["TRAIN"]["DATALOADER"]["MINIBATCH_SIZE_PER_DIVICE"])
        self.model.train()

    def after_epoch(self):
        self.save_ckpt()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            batch_data = next(self.train_loader)
            self.train_one_iter(batch_data)



    def stat_mem(self):
        mem_alloc = torch.cuda.memory_allocated(self.local_rank) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**2
        mem_used = self.gpu_monitor.get_device_mem_info(self.local_rank).used / 1024**2
        mem_info = f"rank:{self.local_rank}, Mem: ({mem_alloc:.2f}/{mem_reserved:.2f}/{mem_used:.2f}) MB"
        print(mem_info)

    def train_one_iter(self, batch_data):
        iter_start_time = time.time()

        lr = get_learning_rate(self.epoch, self.iter, self.base_lr, self.max_iter, self.warmup_epoch, self.max_epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        image = Tensor(batch_data['img']).cuda(self.local_rank).float()
        del batch_data['img']

        for k in batch_data:
            batch_data[k] = Tensor(batch_data[k]).cuda(self.local_rank).float()

        tdata = time.time() - iter_start_time
        
        self.optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            losses = self.model(image, batch_data)
            loss = losses['total_loss']
        self.grad_scaler.scale(loss).backward()
        if self.cfg['TRAIN'].get("CLIP_GRAD", None) is not None:
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['TRAIN']['CLIP_GRAD']['MAX_NORM'])
        else:
            grad_norm = 0.0

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        cur_time = time.time()
        if self.rank == 0 and self.iter % 10 == 0:
            ttrain = cur_time - iter_start_time
            mb_per_second = 1 / (cur_time - iter_start_time)
            result_str = f"rank {self.rank}/ {self.args.world_size}, e: {self.epoch}[{self.iter}/{self.max_iter}], {mb_per_second:.2f}mb/s," 
            result_str += f"lr: {lr :6f}, grad_norm: {grad_norm.item() :.6f}, "

            mem_alloc = torch.cuda.memory_allocated(self.local_rank) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**2
            mem_used = self.gpu_monitor.get_device_mem_info(self.local_rank).used / 1024**2
            mem_info = f"Mem: ({mem_alloc:.2f}/{mem_reserved:.2f}/{mem_used:.2f}) MB"
            result_str += mem_info

            for k,v in losses.items():
                result_str +=f' {k} : {v.item():.5f}, '
            
            if tdata/ttrain > .05:
                result_str += f"dp/tot: {tdata/ttrain:.2g}"
            
            cur_step = self.iter + self.max_iter * self.epoch
            if self.rank == 0:
                self.writer.add_scalar("lr", lr, cur_step)
                for k, v in losses.items():
                    self.writer.add_scalar(f"losses/{k}", v.item(), cur_step)                

            logger.info(result_str)
        dist.barrier()


    def resume_train(self, model):
        self.start_epoch = 0
        if self.args.resume:
            logger.info("resume training")
            ckpt_file = os.path.join(self.log_model_dir, "latest")

            if os.path.exists(ckpt_file):
                loc = 'cuda:{}'.format(self.local_rank)
                checkpoint = torch.load(open(ckpt_file, "rb"), map_location=loc)
                model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.grad_scaler.load_state_dict(checkpoint["scaler_state_dict"])
                start_epoch = checkpoint["epoch"]       

                self.start_epoch = start_epoch
                logger.info(
                    "loaded checkpoint '{}' (epoch {})".format(
                        self.args.resume, self.start_epoch
                    )
            )
        return model

         
    def save_ckpt(self):
        if self.rank == 0:
            logger.info("Dump model begin...")
            model = self.model
            model_to_save = model.module if hasattr(model, "module") else model

            checkpoint = {
                "epoch": self.epoch + 1,
                "state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.grad_scaler.state_dict(),
            }

            if not os.path.exists(self.log_model_dir):
                os.mkdir(self.log_model_dir)

            latest_ckp_path = os.path.join(
                self.log_model_dir, "latest")
            with open(latest_ckp_path, 'wb') as fobj:
                torch.save(checkpoint, fobj)

            if (self.epoch + 1) % self.dump_epoch_interval == 0:
                ckp_path = os.path.join(self.log_model_dir, f"epoch_{self.epoch + 1}")
                with open(ckp_path, 'wb') as fobj:
                    torch.save(checkpoint, fobj)
         
            logger.info("Dump model Done !")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")

    parser.add_argument('--dist_backend', default='nccl',
                        type=str, help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.world_size > 1:
            torch.cuda.manual_seed_all(args.seed)
            
            
    trainer = Trainer(_CONFIG, args)
    trainer.train()
    # trainer.profile()



if __name__ == "__main__":
    main()
