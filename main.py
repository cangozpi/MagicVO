from data import VisualOdometryDataset
import torch
# from models.FlowNet2_src import FlowNet2S
from models.MagicVO_src.MagicVO_model import MagicVO_Model
from models.CNN_Backbone_src.CNN_Backbone_model import CNN_backbone_model
from utils.train import train_with_flownet, train_with_cnn_backbone


def main():
    #TODO: Parse config.yaml and command line args

    # Initialize DataLoaders
    dset_path = "./dataset"
    train_dataset = VisualOdometryDataset(dset_path, 192, 640, ['00'])
    val_dataset = VisualOdometryDataset(dset_path, 192, 640, ['00'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2)
    
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
    
        # Initialize MagicVO model
        magicVO_model = MagicVO_Model()

        # Train the model
        epochs = 5
        lr = 1e-3
        k = 1 #TODO check its value from the paper
        train_flownet = False # Whether to freeze the layers of flownet during training
        magicVO_ckpt_path = "./models/checkpoints/magicVO_best_val_ckpt.pth.tar" # Path to save the magicVO model checkpoints
        flownet_ckpt_path = "./models/checkpoints/flownet_best_val_ckpt.pth.tar" # Path to save the flownet model checkpoints
        load_flownet_ckpt = False # Load flownet from a ckpt saved during training
        load_magicVO_ckpt = False # Load magicvo from a ckpt saved during training
        train_with_flownet(flownet2_model, magicVO_model, train_dataloader, val_dataloader, epochs, lr, k, flownet_ckpt_path, magicVO_ckpt_path,\
            train_flownet=train_flownet, load_flownet_ckpt=load_flownet_ckpt, load_magicVO_ckpt=load_magicVO_ckpt)
    else: # Use CNN_Backbone as the feature extractor
        # Initialize CNN_Backbone_model 
        cnn_backbone_model = CNN_backbone_model()
        # Initialize MagicVO model
        magicVO_model = MagicVO_Model()

        # Train the model
        epochs = 5
        lr = 1e-3
        k = 1 #TODO check its value from the paper
        magicVO_ckpt_path = "./models/checkpoints/magicVO_best_val_ckpt.pth.tar" # Path to save the magicVO model checkpoints
        cnn_backbone_ckpt_path = "./models/checkpoints/flownet_best_val_ckpt.pth.tar" # Path to save the cnn_backbone model checkpoints
        load_cnn_backone_ckpt = False # Load cnn_backbone from a ckpt saved during training
        load_magicVO_ckpt = False # Load magicvo from a ckpt saved during training
        train_with_cnn_backbone(cnn_backbone_model, magicVO_model, train_dataloader, val_dataloader, epochs, lr, k, cnn_backbone_ckpt_path, magicVO_ckpt_path,\
            load_cnn_backone_ckpt=load_cnn_backone_ckpt, load_magicVO_ckpt=load_magicVO_ckpt)


if __name__ == "__main__":
    main()