# MagicVO
Implementation of the [_MagicVO: End-to-End Monocular Visual Odometry through Deep Bi-directional Recurrent Convolutional Neural Network_](https://arxiv.org/abs/1811.10964) paper in pytorch.

---
## Installing Python Packages:
* Using conda Environments:
    ```bash
    $ conda create --name <env> --file requirements.txt
    ```
* Using pip to install:
    ```bash
    $ python -m pip install -r requirements.txt

---

## Installation (FlowNet2)
1. 
    ```bash
    # install custom layers
    $ cd FlowNet2_src
    $ bash install.sh
    ```

2. If you are going to use __FlowNet__ model, you you need to uncomment line 2 of main.py:
    ```python
    # from models.FlowNet2_src import FlowNet2S
    ```

<u>_\# Note that if you are __not__ going to use __FlowNet__ model, you can skip this step and use the alternative __CNN architecture__._</u>

---

## To Run:
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

## Code Structure:
* __main.py__: Calls train/test scripts with the parsed configurations.
* __data.py__: Constructs pytorch Datasets for KITTI Odometry dataset.
* __config.yaml__: Configurations/hyper-parameters for the training/testing.
* __utils/__ -->
    * __train.py__: Implements model training functionality.
    * __test.py__: Implements model testing functionality.
    * __hellpers.py__: Implements helper functions for parsing the command line arguments.
* __results__: Training/Testing results/plots are saved to this folder.
* __models/__ -->
    * __CNN_Backbone_src__:
        * __CNN_Backbone_model.py__: Implements the CNN Architecture which is illustrated in the _MagicVO_ paper as a pytorch Module. This architecture is an available alternative to using the _FlowNet_ model.
    * __FlowNet2_src__: Contains the implemenation of the _FlowNet_ model in pytorch. This source code is taken from https://github.com/vt-vl-lab/flownet2.pytorch. This architecture is an available alternative to using the _CNN architecture_.
        * __install.sh__: This bash script needs to be run in order to install custom layers used by the _FlowNet_ model.
        * __...__: Many other files/folders required by the _FlowNet_ implementation. 
    * __MagicVO_src/__ -->
        * __MagicVO_model.py__: Implements the BI-LSTM+MLP architecture explained in the _MagicVO_ paper as a pytorch Module.
    * __checkpoints__: This is the directory where the saved/to be loaded model checkpoints for all of the models are saved.
    * __dataset__: This folder corresponds to the KITTI Odometry dataset. Simply replace this folder with the KITTI Odometry dataset you've downloaded on your computer by using the same name.
* __requirements.txt__: This file can be used with pip/conda to install the python packages which were used during the development of this code. 

---

## __TODOs:__
1. DataLoaders. :heavy_check_mark:
2. Pre-trained FlowNet model :heavy_check_mark:
3. CNN architecture as an alternative to FlowNet model :heavy_check_mark:
4. MagicVO Implementation :heavy_check_mark:
5. Training/Validation Loop :heavy_check_mark:
6. Saving/Loading trained models :heavy_check_mark:
7. Fix install.sh issue of Pre-trained FlowNet model (only for CPU) :warning:
8. Testing of the model :heavy_check_mark:
9. Utility functions for the visualization of the test results :heavy_check_mark:
10. Configurations yaml file and command line arguments parsing :heavy_check_mark:
11. Add Data Augmentation
12. Add Attention based architecture implementation alternative to the bi-lstm MagicVO model.
13. GPU training/Final Results 

---

## References:
* DataLoader&Test result Visualizations adapted from https://github.com/EduardoTayupanta/VisualOdometry

* FlowNet2 Model adapted from https://github.com/vt-vl-lab/flownet2.pytorch
    ```
    @InProceedings{IMKDB17,
    author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
    title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
    booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    month        = "Jul",
    year         = "2017",
    url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
    }
    ```

---

