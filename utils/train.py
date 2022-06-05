import torch
import matplotlib.pyplot as plt

def plot_results(train_losses, val_losses):
    plt.figure()
    plt.plot(list(range(1, len(train_losses)+1)), train_losses, marker='o')
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/Training Losses Plot")

    plt.figure()
    plt.plot(list(range(1, len(val_losses)+1)), val_losses, marker='o')
    plt.title("Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/Validation Losses Plot")

    plt.show()

def train_with_flownet(flownet_model, magicVO_model, train_dataloader, val_dataloader, epochs, lr, k, flownet_ckpt_path, magicVO_ckpt_path,\
     train_flownet=False, load_flownet_ckpt=False, load_magicVO_ckpt=False):
    optim_flownet = torch.optim.Adagrad(flownet_model.parameters(), lr)
    optim_magicVO = torch.optim.Adagrad(magicVO_model.parameters(), lr)

    # ======== Load models from ckpts
    loaded_epoch = 0
    loaded_loss = None
    # Load magicVO_model & optimizer from ckpt
    if load_flownet_ckpt:
        checkpoint = torch.load(flownet_ckpt_path)
        flownet_model.load_state_dict(checkpoint['flownet_model_state_dict'])
        optim_flownet.load_state_dict(checkpoint['flownet_optimizer_state_dict'])
        loaded_epoch = checkpoint['epoch']
        loaded_loss = checkpoint['loss']
        print(f"Loaded flownet_model from saved ckpt. \tloaded epoch:{loaded_epoch}, \t loaded best validation loss: {loaded_loss}")
    
    # Load flownet_model & optimizer from ckpt
    if load_magicVO_ckpt:
        checkpoint = torch.load(magicVO_ckpt_path)
        magicVO_model.load_state_dict(checkpoint['magicVO_model_state_dict'])
        optim_magicVO.load_state_dict(checkpoint['magicVO_optimizer_state_dict'])
        loaded_epoch = checkpoint['epoch']
        loaded_loss = checkpoint['loss']
        print(f"Loaded magicVO_model from saved ckpt. \tloaded epoch:{loaded_epoch}, \t loaded best validation loss: {loaded_loss}")
    


    # ======== Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flownet_model = flownet_model.to(device)
    magicVO_model = magicVO_model.to(device)

    if device != "cpu":
        print('=' * 30)
        print('Found device:', torch.cuda.get_device_name(0))
        print('=' * 30)
    
    print('=' * 30)
    print('Training MagicVO model')
    print('=' * 30)

    train_losses = [] # i^th index holds the avg training loss for the i^th epoch
    val_losses = [] # i^th index holds the avg validation loss for the i^th epoch
    best_val_loss = None if loaded_loss == None else loaded_loss # best validation loss (used during saving checkpoints)
    # Train/Validate the model
    for i in range(epochs - loaded_epoch): # train for #epochs
        # ======== Train
        magicVO_model.train()
        cum_train_loss = 0
        for step, (img_cat, odometry) in enumerate(train_dataloader): # per batch training
            img_cat = img_cat.to(device)
            odometry = odometry.to(device)
            # Note that FlowNetS requires inputs as [B, 3(RGB), 2(pair), H, W] so reshape the image
            # img_cat = [BX 3X2 (RGBXpair), H , W]
            img_cat = img_cat.view(img_cat.shape[0], 3, 2, img_cat.shape[-2], img_cat.shape[-1])
            
            optim_flownet.zero_grad()
            optim_magicVO.zero_grad()

            train_loss = 0
            if train_flownet == False: # FlowNet model is not trainable
                flownet_model.eval()
                # Extract image features using FlowNet/CNNs
                with torch.no_grad():
                    out = flownet_model(img_cat)
                # make 6 DoF predictions using MagicVO_model
                train_loss = magicVO_model.loss(out, odometry, k)
                
                # Backpropagate the gradients
                train_loss.backward()
                optim_magicVO.step()

            else: # FlowNet model is trainable
                flownet_model.train()
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

        
        # ======== Validate
        flownet_model.eval()
        magicVO_model.eval()
        cum_val_loss = 0
        for img_cat, odometry in val_dataloader: # per batch evaluation
            img_cat = img_cat.to(device)
            odometry = odometry.to(device)
            # Note that FlowNetS requires inputs as [B, 3(RGB), 2(pair), H, W] so reshape the image
            # img_cat = [BX 3X2 (RGBXpair), H , W]
            img_cat = img_cat.view(img_cat.shape[0], 3, 2, img_cat.shape[-2], img_cat.shape[-1])
            
            with torch.no_grad():
                # Extract image features using FlowNet/CNNs
                out = flownet_model(img_cat)
                # make 6 DoF predictions using MagicVO_model
                val_loss = magicVO_model.loss(out, odometry, k)

                
                cum_val_loss += val_loss.item() * len(img_cat) # convert average loss back to total loss
        
        # Record the validation loss
        avg_val_loss = cum_val_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)


        # ======== Save Checkpoints
        if (best_val_loss == None) or (best_val_loss >= avg_val_loss):
            best_val_loss = avg_val_loss
            # Save flownet
            torch.save({
                'epoch': i,
                'flownet_model_state_dict': flownet_model.state_dict(),
                'flownet_optimizer_state_dict': optim_flownet.state_dict(),
                'loss': avg_val_loss,
            }, flownet_ckpt_path)
            # Save MagicVO
            torch.save({
                'epoch': i,
                'magicVO_model_state_dict': magicVO_model.state_dict(),
                'magicVO_optimizer_state_dict': optim_magicVO.state_dict(),
                'loss': avg_val_loss,
            }, magicVO_ckpt_path)
            print("*" * 20)
            print(f"Saved model checkpoint at epoch: {i+loaded_epoch}, \tTraining Loss: {avg_train_loss},  \twith Validation Loss: {avg_val_loss}")
        else:
             print(f'] Epoch: {i+loaded_epoch}/{epochs}, \tTraining Loss: {avg_train_loss}, \tValidation Loss: {avg_val_loss}')

    # ======== Plot Training Results
    plot_results(train_losses, val_losses)



def train_with_cnn_backbone(cnn_backbone_model, magicVO_model, train_dataloader, val_dataloader, epochs, lr, k, cnn_backbone_ckpt_path, magicVO_ckpt_path,\
            load_cnn_backone_ckpt=False, load_magicVO_ckpt=False):
    optim_cnn_backbone = torch.optim.Adagrad(cnn_backbone_model.parameters(), lr)
    optim_magicVO = torch.optim.Adagrad(magicVO_model.parameters(), lr)

    # ======== Load models from ckpts
    loaded_epoch = 0
    loaded_loss = None
    # Load magicVO_model & optimizer from ckpt
    if load_cnn_backone_ckpt:
        checkpoint = torch.load(cnn_backbone_ckpt_path)
        cnn_backbone_model.load_state_dict(checkpoint['cnn_backbone_model_state_dict'])
        optim_cnn_backbone.load_state_dict(checkpoint['cnn_backbone_optimizer_state_dict'])
        loaded_epoch = checkpoint['epoch']
        loaded_loss = checkpoint['loss']
        print(f"Loaded cnn_backbone_model from saved ckpt. Loaded epoch:{loaded_epoch}, Loaded best validation loss: {loaded_loss}")
    
    # Load cnn_backbone_model & optimizer from ckpt
    if load_magicVO_ckpt:
        checkpoint = torch.load(magicVO_ckpt_path)
        magicVO_model.load_state_dict(checkpoint['magicVO_model_state_dict'])
        optim_magicVO.load_state_dict(checkpoint['magicVO_optimizer_state_dict'])
        loaded_epoch = checkpoint['epoch']
        loaded_loss = checkpoint['loss']
        print(f"Loaded magicVO_model from saved ckpt. Loaded epoch:{loaded_epoch}, Loaded best validation loss: {loaded_loss}")
    


    # ======== Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_backbone_model = cnn_backbone_model.to(device)
    magicVO_model = magicVO_model.to(device)

    if device != "cpu":
        print('=' * 30)
        print('Found device:', torch.cuda.get_device_name(0))
        print('=' * 30)

    print('=' * 30)
    print('Training MagicVO model')
    print('=' * 30)

    train_losses = [] # i^th index holds the avg training loss for the i^th epoch
    val_losses = [] # i^th index holds the avg validation loss for the i^th epoch
    best_val_loss = None if loaded_loss == None else loaded_loss # best validation loss (used during saving checkpoints)
    # Train/Validate the model
    for i in range(epochs- loaded_epoch): # train for #epochs
        # ======== Train
        cnn_backbone_model.train()
        magicVO_model.train()
        cum_train_loss = 0
        for step, (img_cat, odometry) in enumerate(train_dataloader): # per batch training
            img_cat = img_cat.to(device)
            odometry = odometry.to(device)
            
            optim_cnn_backbone.zero_grad()
            optim_magicVO.zero_grad()

            # Extract image features using FlowNet/CNNs
            out = cnn_backbone_model(img_cat)
            # make 6 DoF predictions using MagicVO_model
            train_loss = magicVO_model.loss(out, odometry, k)

            # Backpropagate the gradients
            train_loss.backward()
            optim_cnn_backbone.step()
            optim_magicVO.step()


            
            cum_train_loss += train_loss.item() * len(img_cat) # convert average loss back to total loss
        
        # Record the training loss
        avg_train_loss = cum_train_loss / len(train_dataloader.dataset)
        train_losses.append(avg_train_loss)

        
        # ======== Validate
        cnn_backbone_model.eval()
        magicVO_model.eval()
        cum_val_loss = 0
        for img_cat, odometry in val_dataloader: # per batch evaluation
            img_cat = img_cat.to(device)
            odometry = odometry.to(device)
            
            with torch.no_grad():
                # Extract image features using FlowNet/CNNs
                out = cnn_backbone_model(img_cat)
                # make 6 DoF predictions using MagicVO_model
                val_loss = magicVO_model.loss(out, odometry, k)

                
                cum_val_loss += val_loss.item() * len(img_cat) # convert average loss back to total loss
        
        # Record the validation loss
        avg_val_loss = cum_val_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)


        # ======== Save Checkpoints
        if (best_val_loss == None) or (best_val_loss >= avg_val_loss):
            best_val_loss = avg_val_loss
            # Save cnn_backbone
            torch.save({
                'epoch': i+loaded_epoch,
                'cnn_backbone_model_state_dict': cnn_backbone_model.state_dict(),
                'cnn_backbone_optimizer_state_dict': optim_cnn_backbone.state_dict(),
                'loss': avg_val_loss,
            }, cnn_backbone_ckpt_path)
            # Save MagicVO
            torch.save({
                'epoch': i+loaded_epoch,
                'magicVO_model_state_dict': magicVO_model.state_dict(),
                'magicVO_optimizer_state_dict': optim_magicVO.state_dict(),
                'loss': avg_val_loss,
            }, magicVO_ckpt_path)
            print("*" * 20)
            print(f"Saved model checkpoint at epoch: {i+loaded_epoch}, \tTraining Loss: {avg_train_loss},  \twith Validation Loss: {avg_val_loss}")
        else:
             print(f'] Epoch: {i+loaded_epoch}/{epochs}, \tTraining Loss: {avg_train_loss}, \tValidation Loss: {avg_val_loss}')

    # ======== Plot Training Results
    plot_results(train_losses, val_losses)