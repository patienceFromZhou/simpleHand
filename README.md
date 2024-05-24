# News

**20240521**: 📢✨ Update checkpoints and train logs in download link. Fix some bugs. 

**20240507**: 📢✨ code released. 

**20240306**: 🔥🔥 Our [project page](https://simplehand.github.io/) is available. 🚀🚀

**20240228**: 🎉📄 Paper accepted by CVPR 2024.📄🎉

**20230930**: 🥇🥇 We won the 1st place in Egocentric 3D Hand Pose Estimation challenge.🏆🏆 [[Technical Report]](https://arxiv.org/abs/2310.04769)

# simpleHand

**[JIIOV Technology](https://jiiov.com/)**

**A Simple Baseline for Efficient Hand Mesh Reconstruction**

Zhishan Zhou, Shihao Zhou, Zhi Lv, Minqiang Zou, Tong Wu, Mochen Yu, Yao Tang, Jiajun Liang

[[`Paper`]](https://arxiv.org/pdf/2403.01813.pdf) [`[Project]`](#getting-started)

![framework](images/FPS-PA-MPJPE.png)

**A Simple Baseline for Efficient Hand Mesh Reconstruction (simpleHand)** has been accepted by CVPR2024.  This paper ropose a simple yet effective baseline that not only surpasses state-of-the-art (SOTA) methods but also demonstrates computational efficiency. SimpleHand can be easily transplant to mainstream backbones and datasets.

# Getting Started
## Installation
First you need to clone the repo:
```
git clone --recursive https://github.com/patienceFromZhou/simpleHand.git
cd simpleHand
```

We recommend creating a virtual environment for simpleHand. You can use venv:
```bash
python3.10 -m venv .simpleHand
source .simpleHand/bin/activate
```

or alternatively conda:
```bash
conda create --name simpleHand python=3.10
conda activate simpleHand
```

Then install the rest of the dependencies by
```bash
pip3 install -r requirement.txt
```


## Training

Please visit the [FreiHAND project website](https://lmb.informatik.uni-freiburg.de/projects/freihand/) to download FreiHAND data. Then refer to [FreiHAND toolbox](https://github.com/lmb-freiburg/freihand) for MANO model and generate vertices annotations. Name the annotation files to have the same prefix as the corresponding images and put them in a same folder like:
```
{dataset_dir}
├── 00000000.jpg
├── 00000000.json
├── ...
├── ...
├── 00130239.jpg
├── 00130239.json
```
where the annotation json file is formatted as:
```
dict(
    xyz: List(np.array) # 21x3 
    uv: List(np.array) # 21x2 
    K: List(np.array) # 3x3 
    vertices: List(np.array) # 778x3
    image_path: string # *.jpg
)
```
[***RECOMMENDED***] you can alternatively download the pre-generated images and annatations from google drive. This file can be used directly for training and evaluation, without any additional processing. 
```
wget https://drive.google.com/drive/folders/1BfHjNjxQj3MdsGoq5irCrOskyCA9a64l?usp=drive_link
```
The folder consists of three train files, train.json, eval.json, FreiHAND.zip. Json files specify image paths. ZIP file consists images and annotations. Validate FreiHAND.zip by
```
md5sum FreiHAND.zip
1d58ff7d6029c8ff724471e06803afa4  FreiHAND.zip
```
We release two checkpoints epoch_200_rerun1, epoch_200_rerun2. One can use them for quick comparison or model variance assessment. Also, We provided the out.log, which contains the output from the training process.

Specify the folders in cfg.py. Then you can start training using the following command:
```
make train
```
Checkpoints and logs will be saved to `./train_logs/`.

## evaluation
when training is Done, evaluate the default epoch_200 by
```
make eval
```
here is an example output
```
Evaluation 3D KP results:
auc=0.000, mean_kp3d_avg=70.70 cm
Evaluation 3D KP ALIGNED results:
auc=0.887, mean_kp3d_avg=0.57 cm

Evaluation 3D MESH results:
auc=0.000, mean_kp3d_avg=70.69 cm
Evaluation 3D MESH ALIGNED results:
auc=0.881, mean_kp3d_avg=0.60 cm

F-scores
F@5.0mm = 0.0000        F_aligned@5.0mm = 0.7717
F@15.0mm = 0.0000       F_aligned@15.0mm = 0.9858
```
