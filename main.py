from data import VisualOdometryDataset
import torch
from models.FlowNet2_src import FlowNet2S
from models.MagicVO_src import MagicVO_model

def train(flownet2_model, magicVO_model, train_dataloader, val_dataloader, epochs):
    device = "gpu" if torch.cuda.is_available() else "cpu"
    flownet2_model = flownet2_model.to(device)
    magicVO_model = magicVO_model.to(device)

    # Train the model
    for i in range(epochs): # train for #epochs
        for img_cat, odometry in train_dataloader: # per batch training
            img_cat = img_cat.to(device)
            odometry = odometry.to(device)

            

        
    #TODO: Validate the model
    # for img_cat, odometry in val_dataloader: # per batch evaluation
    #     img_cat = img_cat.to(device)
    #     odometry = odometry.to(device)
    #     pass

    #TODO: save checkpoint/model
    

def main():
    #TODO: Parse config.yaml and command line args

    # Initialize DataLoaders
    path = "/home/cangozpi/Desktop/Docker_shared/Computer Vision for Autonomous Driving/MagicVO/dataset"
    train_dataset = VisualOdometryDataset(path, 192, 640, ['00'])
    val_dataset = VisualOdometryDataset(path, 192, 640, ['00'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)
    
    # Load pre-trained FlowNet model
    flownet2_model = FlowNet2S()
    flownet_path = 'models/checkpoints/FlowNet2-S_checkpoint.pth.tar'
    pretrained_dict = torch.load(flownet_path)['state_dict']
    model_dict = flownet2_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    flownet2_model.load_state_dict(model_dict)

    # Initialize MagicVO model
    magicVO_model = MagicVO_model()

    # Train the model
    epochs = 5
    train(flownet2_model, magicVO_model, train_dataloader, val_dataloader, epochs)



if __name__ == "__main__":
    main()