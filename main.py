from data import VisualOdometryDataset
import torch
from models.FlowNet2_src import FlowNet2S
from models.MagicVO_src import MagicVO_model

def train(flownet_model, magicVO_model, train_dataloader, val_dataloader, epochs, lr, k, train_flownet=False):
    device = "gpu" if torch.cuda.is_available() else "cpu"
    flownet_model = flownet_model.to(device)
    magicVO_model = magicVO_model.to(device)

    optim_flownet = torch.optim.Adagrad(flownet_model.parameters(), lr)
    optim_magicVO = torch.optim.Adagrad(magicVO_model.parameters(), lr)

    print('=' * 30)
    print('Training MagicVO model')
    print('=' * 30)

    train_losses = [] # i^th index holds the avg training loss for the i^th epoch
    val_losses = [] # i^th index holds the avg validation loss for the i^th epoch
    # Train the model
    for i in range(epochs): # train for #epochs
        cum_train_loss = 0
        for step, (img_cat, odometry) in enumerate(train_dataloader): # per batch training
            img_cat = img_cat.to(device)
            odometry = odometry.to(device)
            
            optim_flownet.zero_grad()
            optim_magicVO.zero_grad()

            train_loss = 0
            if train_flownet == False: # FlowNet model is not trainable
                # Extract image features using FlowNet/CNNs
                with torch.no_grad():
                    out = flownet_model(img_cat)
                # make 6 DoF predictions using MagicVO_model
                train_loss = magicVO_model.loss(out, odometry, k)
                
                # Backpropagate the gradients
                train_loss.backward()
                optim_magicVO.step()

            else: # FlowNet model is trainable
                # Extract image features using FlowNet/CNNs
                out = flownet_model(img_cat)
                # make 6 DoF predictions using MagicVO_model
                train_loss = magicVO_model.loss(out, odometry, k)

                # Backpropagate the gradients
                train_loss.backward()
                optim_flownet.step()
                optim_magicVO.step()


            
            cum_train_loss += train_loss.item() * len(img_cat) # convert average loss back to total loss
        
        # Record the training loss
        avg_train_loss = cum_train_loss / len(train_dataloader.dataset)
        train_losses.append(avg_train_loss)

        
        cum_val_loss = 0
        for img_cat, odometry in val_dataloader: # per batch evaluation
            img_cat = img_cat.to(device)
            odometry = odometry.to(device)
            
            with torch.no_grad():
                # Extract image features using FlowNet/CNNs
                out = flownet_model(img_cat)
                # make 6 DoF predictions using MagicVO_model
                val_loss = magicVO_model.loss(out, odometry, k)

                
                cum_val_loss += val_loss.item() * len(img_cat) # convert average loss back to total loss
        
        # Record the validation loss
        avg_val_loss = cum_val_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)


        #TODO: save checkpoint/model
        
    

def main():
    #TODO: Parse config.yaml and command line args

    # Initialize DataLoaders
    path = "/home/cangozpi/Desktop/Docker_shared/Computer Vision for Autonomous Driving/MagicVO/dataset"
    train_dataset = VisualOdometryDataset(path, 192, 640, ['00'])
    val_dataset = VisualOdometryDataset(path, 192, 640, ['00'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2)
    
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
    lr = 1e-3
    k = 1 #TODO check its value from the paper
    train(flownet2_model, magicVO_model, train_dataloader, val_dataloader, epochs, lr, k, train_flownet=False)



if __name__ == "__main__":
    main()