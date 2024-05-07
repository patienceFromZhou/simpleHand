import pynvml
from pathlib import Path
from tempfile import TemporaryDirectory
import stat
from typing import List

        
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def make_symlink_if_not_exists(src: Path, dst: Path, overwrite=False):
    """:param override: if destination exists and it is already a symlink
    then overwrite it."""

    src, dst = Path(src), Path(dst)

    if not overwrite:
        dst.symlink_to(src)
        return

    while True:
        try:
            s = dst.lstat()
            break
        except FileNotFoundError:
            try:
                dst.symlink_to(src)
                return
            except FileExistsError:
                continue

    if not stat.S_ISLNK(s.st_mode):
        raise FileExistsError("{} exists and is not a symlink".format(dst))

    with TemporaryDirectory(dir=str(dst.parent)) as tmpdir:
        tmplink = Path(tmpdir, "x")
        tmplink.symlink_to(src)
        tmplink.rename(dst)


class GPUMemoryMonitor:

    def __init__(self) -> None:
        pynvml.nvmlInit()
        self.ndevices = pynvml.nvmlDeviceGetCount()

    def get_memory_info(self) -> List[pynvml.c_nvmlMemory_t]:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.ndevices)]
        return [pynvml.nvmlDeviceGetMemoryInfo(handle) for handle in handles]

    def get_device_mem_info(self, device=0) -> pynvml.c_nvmlMemory_t:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle)

    def __delete__(self):
        pynvml.nvmlShutdown()    
        

def get_log_model_dir(tag=""):
    exp_name = "models"
    if len(tag):
        exp_name = f"{exp_name}_{tag}"
    model_dir = os.path.join("train_log", exp_name)
    return model_dir