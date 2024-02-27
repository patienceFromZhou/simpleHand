# simpleHand
**[Jiiov Technology](https://jiiov.com/)**

**A Simple Baseline for Efficient Hand Mesh Reconstruction**
Zhishan Zhou, Shihao Zhou, Zhi Lv, Minqiang Zou, Tong Wu, Mochen Yu, Yao Tang, Jiajun Liang

[`Paper`] [[`Project`](https://github.com/patienceFromZhou/simpleHand?tab=readme-ov-file)]

![framework](images/overview.png)

**A Simple Baseline for Efficient Hand Mesh Reconstruction (simpleHand)**    propose a simple yet effective baseline that not only surpasses state-of-the-art (SOTA) methods but also demonstrates computational efficiency. SimpleHand can be easily transplant to mainstream backbones and datasets.

![framework](images/framework.png)

 SimpleHand is abstracted into a token generator and a mesh regressor. Token generator samples representative tokens using predicted 2d keypoints. Mesh regressor cascadely lifts the sampled tokens into meshes.

![framework](images/comparison.png)
SimpleHand capitalizes on the strengths of existing methodologies, thereby outperforming them in numerous challenging scenarios. This is particularly evident in intricate finger interactions, like pinching or twisting, as well as in complex and unconventional gestures.