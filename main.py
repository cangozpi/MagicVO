from data import VisualOdometryDataset
import torch
from models.FlowNet2_src import FlowNet2S



def main():
    #TODO: Parse config.yaml and command line args

    # Initialize DataLoaders
    path = "/home/cangozpi/Desktop/Docker_shared/Computer Vision for Autonomous Driving/MagicVO/dataset"
    train_dataset = VisualOdometryDataset(path, 192, 640, ['00'])
    val_dataset = VisualOdometryDataset(path, 192, 640, ['00'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)
    
    # Load pre-trained FlowNet model
    flownet2 = FlowNet2S()
    flownet_path = 'models/checkpoints/FlowNet2-S_checkpoint.pth.tar'
    pretrained_dict = torch.load(flownet_path)['state_dict']
    model_dict = flownet2.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    flownet2.load_state_dict(model_dict)
    


if __name__ == "__main__":
    main()