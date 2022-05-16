# MagicVO
Implementation of the [_MagicVO: End-to-End Monocular Visual Odometry through Deep Bi-directional Recurrent Convolutional Neural Network_](https://arxiv.org/abs/1811.10964) paper in pytorch.

---
## Installing Python Packages:
* Using conda environment:
    ```bash
    $ conda create --name <env> --file requirements.txt
    ```
* Using pip to install:
    ```bash
    $ python -m pip install -r requirements.txt

---

## Converted Caffe Pre-trained Models
* [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]
* [FlowNet2-C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[149MB]
* [FlowNet2-CS](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)[297MB]
* [FlowNet2-CSS](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)[445MB]
* [FlowNet2-CSS-ft-sd](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)[445MB]
* [FlowNet2-S](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)[148MB]
* [FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]

_# Note that the default code uses __FlowNet2-S__ weights._

---

## To Run:
* To run using the configurations set in the __config.yaml__ file. 
    ```bash
        $ python main.py
    ```
    _#Note: Check out __config.yaml__ to see with which configurations you are runnign the code with. Code behaviour such as training/testing, model hyperparameters and much more can be configured using this file. Please refer to the comments in this file to understand what each parameter does._

_or_

* To run using the configurations set in the specified yaml file. 
    ```bash
        $ python main.py --config_path="<configurations yaml file path>"
    ```

---

## Code Structure:
* __main.py__: Calls train/test scripts with the parsed configurations.
* __data.py__: Constructs pytorch Datasets for KITTI Odometry dataset.
* __config.yaml__: Configurations/hyper-parameters for the training/testing. Check the comments in this file to modify code behaviour.
* __utils/__ -->
    * __train.py__: Implements model training functionality.
    * __test.py__: Implements model testing functionality.
    * __hellpers.py__: Implements helper functions for parsing the command line arguments.
* __results__: Training/Testing results/plots are saved to this folder.
* __models/__ -->
    * __CNN_Backbone_src__ -->
        * __CNN_Backbone_model.py__: Implements the CNN Architecture which is illustrated in the _MagicVO_ paper as a pytorch Module. This architecture is an available alternative to using the _FlowNet_ model.
    * __FlowNet2_src__ --> Contains the implemenation of the _FlowNet_ model in pytorch. This source code is taken/modified from https://github.com/vt-vl-lab/flownet2.pytorch. This architecture is an available alternative to using the _CNN architecture_.
        * __...__: Many other files/folders required by the _FlowNet_ implementation. 
    * __MagicVO_src/__ -->
        * __MagicVO_model.py__: Implements the BI-LSTM+MLP architecture explained in the _MagicVO_ paper as a pytorch Module.
    * __checkpoints__: This is the directory where the saved/to be loaded model checkpoints for all of the models are saved.
* __dataset__: This folder corresponds to the KITTI Odometry dataset. Simply replace this folder with the KITTI Odometry dataset you've downloaded on your computer by using the same name. In other words this folder should have "__poses__" and "__sequences__" folders inside.
* __requirements.txt__: This file can be used with pip/conda to install the python packages which were used during the development of this code. 

---

## __TODOs:__
1. DataLoaders. :heavy_check_mark:
2. Pre-trained FlowNet model :heavy_check_mark:
3. CNN architecture as an alternative to FlowNet model :heavy_check_mark:
4. MagicVO Implementation :heavy_check_mark:
5. Training/Validation Loop :heavy_check_mark:
6. Saving/Loading trained models :heavy_check_mark:
7. Fix install.sh issue of Pre-trained FlowNet model (only for CPU) :heavy_check_mark:
8. Testing of the model :heavy_check_mark:
9. Utility functions for the visualization of the test results :heavy_check_mark:
10. Configurations yaml file and command line arguments parsing :heavy_check_mark:
11. Initial GPU training
12. Add Data Augmentation
13. Add Attention based architecture implementation alternative to the bi-lstm MagicVO model.
14. Final GPU training/sharing results 

---

## References:
* DataLoader&Test result Visualizations adapted from https://github.com/EduardoTayupanta/VisualOdometry

* FlowNet2 Model adapted from https://github.com/vt-vl-lab/flownet2.pytorch. Links for the pre-trained Caffe weights belong here.
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
* [___MagicVO__: End-to-End Monocular Visual Odometry through Deep Bi-directional Recurrent Convolutional Neural Network_](https://arxiv.org/abs/1811.10964)
    ```
    @misc{https://doi.org/10.48550/arxiv.1811.10964,
    doi = {10.48550/ARXIV.1811.10964},
    
    url = {https://arxiv.org/abs/1811.10964},
    
    author = {Jiao, Jian and Jiao, Jichao and Mo, Yaokai and Liu, Weilun and Deng, Zhongliang},
    
    keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
    
    title = {MagicVO: End-to-End Monocular Visual Odometry through Deep Bi-directional Recurrent Convolutional Neural Network},
    
    publisher = {arXiv},
    
    year = {2018},
    
    copyright = {arXiv.org perpetual, non-exclusive license}
    }
    ```

---

