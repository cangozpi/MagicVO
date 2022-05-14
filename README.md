# MagicVO
Implementation of MagicVO: End-to-End Monocular Visual Odometry through Deep Bi-directional  Recurrent Convolutional Neural Network.

---

## __TODOs:__
1. DataLoaders. :heavy_check_mark:
2. Pre-trained FlowNet model :heavy_check_mark:
3. CNN architecture as an alternative to FlowNet model :heavy_check_mark:
4. MagicVO Implementation :heavy_check_mark:
5. Training/Validation Loop :heavy_check_mark:
6. Saving/Loading trained models :heavy_check_mark:
7. Fix install.sh issue of Pre-trained FlowNet model :warning:
8. Testing of the model :heavy_check_mark:
9. Utility functions for the visualization of the test results :heavy_check_mark:
10. Configurations yaml file and command line arguments parsing :heavy_check_mark:
11. Add Data Augmentation
12. Add Attention based architecture implementation alternative to the bi-lstm MagicVO model.
13. GPU training/Final Results 

---

## Installation (FlowNet2)
```bash
# install custom layers
$ cd FlowNet2_src
$ bash install.sh
```
* FlowNet2 Model adapted from https://github.com/vt-vl-lab/flownet2.pytorch

---

* DataLoader&Test result Visualizations adapted from https://github.com/EduardoTayupanta/VisualOdometry

---

## TO Run:
* To run using the configurations set in the config.yaml file. 
    ```bash
        $ python main.py
    ```
_or_

* To run using the configurations set in the specified yaml file. 
    ```bash
        $ python main.py --config_path="<configurations yaml file path>"
    ```

---
