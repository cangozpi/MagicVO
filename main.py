from data import VisualOdometryDataset
import torch
# from models.FlowNet2_src import FlowNet2S
from models.MagicVO_src.MagicVO_model import MagicVO_Model
from models.CNN_Backbone_src.CNN_Backbone_model import CNN_backbone_model
from utils.train import train_with_flownet, train_with_cnn_backbone
from utils.test import test_with_cnn_backbone, test_with_flownet_backbone


def train_mode():
    # Initialize DataLoaders
    train_dataset = VisualOdometryDataset(config["train_dset_path"], config["height"], config["width"], config["sequences"])
    val_dataset = VisualOdometryDataset(config["val_dset_path"], config["height"], config["width"], config["sequences"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"])
    

    if  config["flownet_or_CNN_backbone"]: # Use FlowNet as the feature extractor
        # Load pre-trained FlowNet model
        flownet2_model = FlowNet2S()
        if config["use_pretrained_flownet"]: # Use pretrained flownet model loaded from Caffee converted ckpts
            pretrained_dict = torch.load(config["flownet_path"])['state_dict']
            model_dict = flownet2_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            flownet2_model.load_state_dict(model_dict)
    
        # Initialize MagicVO model
        magicVO_model = MagicVO_Model()

        # Train the model
        train_with_flownet(flownet2_model, magicVO_model, train_dataloader, val_dataloader, config["epochs"], config["lr"], config["k"], \
            config["flownet_ckpt_path"], config["magicVO_ckpt_path"], train_flownet=config["train_flownet"], \
                load_flownet_ckpt=config["load_flownet_ckpt"], load_magicVO_ckpt=config["load_magicVO_ckpt"])
    else: # Use CNN_Backbone as the feature extractor
        # Initialize CNN_Backbone_model 
        cnn_backbone_model = CNN_backbone_model()
        # Initialize MagicVO model
        magicVO_model = MagicVO_Model()

        # Train the model
        train_with_cnn_backbone(cnn_backbone_model, magicVO_model, train_dataloader, val_dataloader, config["epochs"], config["lr"], config["k"], \
            config["cnn_backbone_ckpt_path"], config["magicVO_ckpt_path"], load_cnn_backone_ckpt=config["load_cnn_backone_ckpt"], \
                load_magicVO_ckpt=config["load_magicVO_ckpt"])

def test_mode():
    # Initialize DataLoaders
    dset_path = "./dataset"
    test_dataset = VisualOdometryDataset(dset_path, 192, 640, ['00'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    flownet_or_CNN_backbone = False # if True use flownet, if not use CNN_backbone

    if flownet_or_CNN_backbone: # Use FlowNet as the feature extractor
        # Load pre-trained FlowNet model
        use_pretrained_flownet = True # Whether to use a pretrained flownet model laoded from Caffee converted ckpts
        flownet2_model = FlowNet2S()
        if use_pretrained_flownet: # Use pretrained flownet model loaded from Caffee converted ckpts
            flownet_path = 'models/checkpoints/FlowNet2-S_checkpoint.pth.tar'
            pretrained_dict = torch.load(flownet_path)['state_dict']
            model_dict = flownet2_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            flownet2_model.load_state_dict(model_dict)
            print('*' *30)
            print("Loaded pre-trained flownet_model from caffee ckpts.")
    
        # Initialize MagicVO model
        magicVO_model = MagicVO_Model()

        # Test the model
        magicVO_ckpt_path = "./models/checkpoints/magicVO_best_val_ckpt.pth.tar" # Path to load the magicVO model checkpoints
        flownet_ckpt_path = "./models/checkpoints/flownet_best_val_ckpt.pth.tar" # Path to load the flownet model checkpoints
        load_flownet_ckpt = False # Load flownet from a ckpt saved during training or use pre-trained flownet
        test_with_flownet_backbone(flownet2_model, magicVO_model, test_dataloader, flownet_ckpt_path, magicVO_ckpt_path, load_flownet_ckpt)
    else: # Use CNN_Backbone as the feature extractor
        # Initialize CNN_Backbone_model 
        cnn_backbone_model = CNN_backbone_model()
        # Initialize MagicVO model
        magicVO_model = MagicVO_Model()

        # Test the model
        k = 1 #TODO check its value from the paper
        magicVO_ckpt_path = "./models/checkpoints/magicVO_best_val_ckpt.pth.tar" # Path to load the magicVO model checkpoints
        cnn_backbone_ckpt_path = "./models/checkpoints/flownet_best_val_ckpt.pth.tar" # Path to load the cnn_backbone model checkpoints
        test_with_cnn_backbone(cnn_backbone_model, magicVO_model, test_dataloader, cnn_backbone_ckpt_path, magicVO_ckpt_path)
        

def main():
    #TODO: Parse config.yaml and command line args
    # ======================
    global config
    config = {
        "mode": "train", # determines whether to run test or the training script. valid values are ["train", "test"]
        "train_dset_path": "./dataset", # path of the training dataset
        "val_dset_path": "./dataset", # path of the validation dataset
        "height": 192, # height to crop the loaded images in the dataset
        "width": 640, # width to crop the loaded images in the dataset
        "sequences": ['00'], # sequences to load from kiti dataset
        "batch_size": 2,
        "epochs": 5,
        "lr": 1e-3,
        "k": 1, #TODO check its value from the paper
        "flownet_or_CNN_backbone": False, # if True use flownet, if not use CNN_backbone
        "use_pretrained_flownet": True, # Whether to use a pretrained flownet model laoded from Caffee converted ckpts
        "flownet_path": 'models/checkpoints/FlowNet2-S_checkpoint.pth.tar', # file path of the pre-trained caffee ckpt for flownet
        "train_flownet": False, # Whether to freeze the layers of flownet during training
        "magicVO_ckpt_path": "./models/checkpoints/magicVO_best_val_ckpt.pth.tar", # Path to save/load the magicVO model checkpoints
        "flownet_ckpt_path": "./models/checkpoints/flownet_best_val_ckpt.pth.tar", # Path to save/load the flownet model checkpoints
        "cnn_backbone_ckpt_path": "./models/checkpoints/flownet_best_val_ckpt.pth.tar", # Path to save/load the cnn_backbone model checkpoints
        "load_flownet_ckpt": False, # Load flownet from a ckpt saved during training
        "load_magicVO_ckpt": False, # Load magicvo from a ckpt saved during training
        "load_cnn_backone_ckpt": False, # Load cnn_backbone from a ckpt saved during training
        




    }

    # ============
    assert config["mode"] in ["train", "test"]

    if config["mode"] == "train":
        train_mode()
    else:
        test_mode()


if __name__ == "__main__":
    main()