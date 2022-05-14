# MagicVO
Implementation of MagicVO: End-to-End Monocular Visual Odometry through Deep Bi-directional  Recurrent Convolutional Neural Network.

---

## __TODOs:__
1. DataLoaders. :heavy_check_mark:
2. Pre-trained FlowNet model :heavy_check_mark:
3. MagicVO Implementation :heavy_check_mark:
4. Training/Validation Loop :heavy_check_mark:
5. Saving/Loading trained models :heavy_check_mark:
6. Fix install.sh issue of Pre-trained FlowNet model :warning:
7. Testing of the model :heavy_check_mark:
8. Utility functions for the visualization of the test results :heavy_check_mark:
9. Add Data Augmentation
10. Add Attention based architecture implementation alternative to the bi-lstm MagicVO model.
11. configurations yaml file and command line arguments parsing
12. GPU training/Final Results 

---

## Installation (FlowNet2)
```bash
# install custom layers
$ cd FlowNet2_src
$ bash install.sh
```
* FlowNet2 Model adapted from https://github.com/vt-vl-lab/flownet2.pytorch

---

* DataLoader adapted from https://github.com/EduardoTayupanta/VisualOdometry

---

## TO Run:
```bash
$ python main.py
```

---
