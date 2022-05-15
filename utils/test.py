import torch
import matplotlib.pyplot as plt
import numpy as np
from data import eulerAnglesToRotationMatrix

def test_with_flownet_backbone(flownet2_model, magicVO_model, test_dataloader, flownet_ckpt_path, magicVO_ckpt_path, load_flownet_ckpt):
    # ======== Load models from ckpts
    loaded_epoch = 0
    loaded_loss = None
    
    if load_flownet_ckpt: # if not using pre-trained flownet from caffee, then use ckpt saved during training
        # Load flownet_model from ckpt
        checkpoint = torch.load(flownet_ckpt_path)
        flownet2_model.load_state_dict(checkpoint['flownet_model_state_dict'])
        loaded_epoch = checkpoint['epoch']
        loaded_loss = checkpoint['loss']
        print(f"Loaded flownet_model from saved ckpt. \tloaded epoch:{loaded_epoch}, \t loaded best validation loss: {loaded_loss}")
    

    # Load magicVO_model from ckpt
    checkpoint = torch.load(magicVO_ckpt_path)
    magicVO_model.load_state_dict(checkpoint['magicVO_model_state_dict'])
    loaded_epoch = checkpoint['epoch']
    loaded_loss = checkpoint['loss']
    print(f"Loaded magicVO_model from saved ckpt. Loaded epoch:{loaded_epoch}, Loaded best validation loss: {loaded_loss}")
    print('*' *30)


    # ======== Use GPU if available
    device = "gpu" if torch.cuda.is_available() else "cpu"
    flownet2_model = flownet2_model.to(device)
    magicVO_model = magicVO_model.to(device)



    print('=' * 30)
    print('Testing MagicVO model')
    print('=' * 30)

    # ======== Test

    T = np.eye(4)
    gtT = np.eye(4)
    estimatedCameraTraj = np.empty([len(test_dataloader.dataset) + 1, 3]) # --> [(B*L)+1, 3] # Note that +1 is for initial pose
    gtCameraTraj = np.empty([len(test_dataloader.dataset) + 1, 3])

    # set the starting poin/origin
    estimatedCameraTraj[0] = np.zeros([1, 3])
    gtCameraTraj[0] = np.zeros([1, 3])

    estimatedFrame = 0
    gtFrame = 0

    flownet2_model.eval()
    magicVO_model.eval()
    for i, (img_cat, odometry) in enumerate(test_dataloader): # per batch evaluation
        img_cat = img_cat.to(device)
        odometry = odometry.to(device)
        # Note that FlowNetS requires inputs as [B, 3(RGB), 2(pair), H, W] so reshape the image
        # img_cat = [BX 3X2 (RGBXpair), H , W]
        img_cat = img_cat.view(img_cat.shape[0], 3, 2, img_cat.shape[-2], img_cat.shape[-1])
        
        with torch.no_grad():
            # Extract image features using FlowNet/CNNs
            out = flownet2_model(img_cat)
            # make 6 DoF predictions using MagicVO_model
            preds = magicVO_model(out) # --> [L, 6] 6 DoF predictions
        
        
        
        # ========== Process predictions
        for pred in preds.numpy(): # iterate through the predictions
            R = eulerAnglesToRotationMatrix(pred[3:])
            t = pred[:3].reshape(3, 1)
            T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            T_abs = np.dot(T, T_r)
            # Update the T matrix till now.
            T = T_abs

            # Get the origin of the frame (i+1), ie the camera center
            estimatedCameraTraj[estimatedFrame + 1] = np.transpose(T[0:3, 3])
            estimatedFrame += 1

        
        # ========== Process Ground Truth (gt)
        for gt in odometry.numpy():
            R = eulerAnglesToRotationMatrix(gt[3:])
            t = gt[:3].reshape(3, 1)
            gtT_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            gtT_abs = np.dot(gtT, gtT_r)
            # Update the T matrix till now.
            gtT = gtT_abs

            # Get the origin of the frame (i+1), ie the camera center
            gtCameraTraj[gtFrame + 1] = np.transpose(gtT[0:3, 3])
            gtFrame += 1
        
        # ========== Plot/Save Results
        x_gt = gtCameraTraj[:, 0]
        z_gt = gtCameraTraj[:, 2]

        x_est = estimatedCameraTraj[:, 0]
        z_est = estimatedCameraTraj[:, 2]

        fig, ax = plt.subplots(1)
        ax.plot(x_gt, z_gt, 'c', label="ground truth")
        ax.plot(x_est, z_est, 'm', label="estimated")
        ax.legend()
        plt.savefig(f"results/Test results visualized {i}")
        
        plt.show()



def test_with_cnn_backbone(cnn_backbone_model, magicVO_model, test_dataloader, cnn_backbone_ckpt_path, magicVO_ckpt_path):
    # ======== Load models from ckpts
    loaded_epoch = 0
    loaded_loss = None
    print('*' *30)
    
    # Load cnn_backbone_model from ckpt
    checkpoint = torch.load(cnn_backbone_ckpt_path)
    cnn_backbone_model.load_state_dict(checkpoint['cnn_backbone_model_state_dict'])
    loaded_epoch = checkpoint['epoch']
    loaded_loss = checkpoint['loss']
    print(f"Loaded cnn_backbone_model from saved ckpt. Loaded epoch:{loaded_epoch}, Loaded best validation loss: {loaded_loss}")
    
    # Load magicVO_model from ckpt
    checkpoint = torch.load(magicVO_ckpt_path)
    magicVO_model.load_state_dict(checkpoint['magicVO_model_state_dict'])
    loaded_epoch = checkpoint['epoch']
    loaded_loss = checkpoint['loss']
    print(f"Loaded magicVO_model from saved ckpt. Loaded epoch:{loaded_epoch}, Loaded best validation loss: {loaded_loss}")
    print('*' *30)


    # ======== Use GPU if available
    device = "gpu" if torch.cuda.is_available() else "cpu"
    cnn_backbone_model = cnn_backbone_model.to(device)
    magicVO_model = magicVO_model.to(device)



    print('=' * 30)
    print('Testing MagicVO model')
    print('=' * 30)

    # ======== Test

    T = np.eye(4)
    gtT = np.eye(4)
    estimatedCameraTraj = np.empty([len(test_dataloader.dataset) + 1, 3]) # --> [(B*L)+1, 3] # Note that +1 is for initial pose
    gtCameraTraj = np.empty([len(test_dataloader.dataset) + 1, 3])

    # set the starting poin/origin
    estimatedCameraTraj[0] = np.zeros([1, 3])
    gtCameraTraj[0] = np.zeros([1, 3])

    estimatedFrame = 0
    gtFrame = 0

    cnn_backbone_model.eval()
    magicVO_model.eval()
    for i, (img_cat, odometry) in enumerate(test_dataloader): # per batch evaluation
        img_cat = img_cat.to(device)
        odometry = odometry.to(device)
        
        with torch.no_grad():
            # Extract image features using FlowNet/CNNs
            out = cnn_backbone_model(img_cat)
            # make 6 DoF predictions using MagicVO_model
            preds = magicVO_model(out) # --> [L, 6] 6 DoF predictions
        
        
        
        # ========== Process predictions
        for pred in preds.numpy(): # iterate through the predictions
            R = eulerAnglesToRotationMatrix(pred[3:])
            t = pred[:3].reshape(3, 1)
            T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            T_abs = np.dot(T, T_r)
            # Update the T matrix till now.
            T = T_abs

            # Get the origin of the frame (i+1), ie the camera center
            estimatedCameraTraj[estimatedFrame + 1] = np.transpose(T[0:3, 3])
            estimatedFrame += 1

        
        # ========== Process Ground Truth (gt)
        for gt in odometry.numpy():
            R = eulerAnglesToRotationMatrix(gt[3:])
            t = gt[:3].reshape(3, 1)
            gtT_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            gtT_abs = np.dot(gtT, gtT_r)
            # Update the T matrix till now.
            gtT = gtT_abs

            # Get the origin of the frame (i+1), ie the camera center
            gtCameraTraj[gtFrame + 1] = np.transpose(gtT[0:3, 3])
            gtFrame += 1
        
        # ========== Plot/Save Results
        x_gt = gtCameraTraj[:, 0]
        z_gt = gtCameraTraj[:, 2]

        x_est = estimatedCameraTraj[:, 0]
        z_est = estimatedCameraTraj[:, 2]

        fig, ax = plt.subplots(1)
        ax.plot(x_gt, z_gt, 'c', label="ground truth")
        ax.plot(x_est, z_est, 'm', label="estimated")
        ax.legend()
        plt.savefig(f"results/Test results visualized {i}")
        
        plt.show()





        

    
    