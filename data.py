import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# Helper functions
def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z], dtype=np.float32)

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

class VisualOdometryDataset(Dataset):
    def __init__(self, datapath, height, width, sequences= ['00']):
        self.base_path = datapath
        # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        self.sequences = sequences # sequences used for training

        self.size = 0 # total number of lines (i.e. time_steps) in the dataset (i.e. sum of all sequences used)
        self.sizes = [] # index i holds the length of number of lines (i.e. time_steps) in the i^th line of sequence.txt 
        # Read in poses folder
        self.poses = self.load_poses() # --> [num_sequences, num_lines (i.e. time_steps), number_of_elements_in_the_line]
        self.width = width
        self.height = height
        # Read in sequences folder
        self.images_stacked, self.odometries = self.get_data() # --> [num_squences*num_lines(total_time_steps), 2] , [num_squences*num_lines(total_time_steps), 6 (3 (i.e. (t, angles)) + 3(e.g. [x,y,z] or t) = 6)]
        # Note that images_stacked is a dict which holds the paths for the two images to be concatenated in each time step.

    def __getitem__(self, index):
        # Read in the consecutive images
        img1_path = self.images_stacked[index,0]
        img2_path = self.images_stacked[index,1]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        # Resize images
        resize_transform = transforms.Resize((self.height, self.width))
        img1_resized = resize_transform(img1)
        img2_resized = resize_transform(img2)
        # Convert PIL images to torch.tensor
        pil_to_tensor_transform = transforms.ToTensor()
        img1_tensor = pil_to_tensor_transform(img1_resized) # --> [C (i.e. 3), H, W]
        img2_tensor = pil_to_tensor_transform(img2_resized) # --> [C (i.e. 3), H, W]
        # Concatenate the consecutive images
        img_cat = torch.cat((img1_tensor, img2_tensor), 0) # --> [C*2 (i.e. 6), H, W]
        
        # obtain odometry data (i.e. gt)
        odometry = self.odometries[index] # --> [6] (3 (i.e. (t, angles)) + 3(e.g. [x,y,z] or t) = 6)
        odometry = torch.tensor(odometry)
        
        return img_cat, odometry # --> C*2 (i.e. 6), H, W] , [6] (i.e. 3 (i.e. (t, angles)) + 3 (e.g. [x,y,z] or t) = 6)
        

    def __len__(self):
        return self.size - len(self.sequences)

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/', sequence + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)

                self.size = self.size + len(poses)
                self.sizes.append(len(poses))
        return all_poses

    def get_image_paths(self, sequence, index):
        image_path = os.path.join(self.base_path, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
        return image_path

    def matrix_rt(self, p):
        # Note that each row of the file contains the first 3 rows of a 4x4 homogeneous pose matrix flattened into one line.
        return np.vstack([np.reshape(p.astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])

    def get_data(self):
        images_paths = []
        odometries = []
        for index, sequence in enumerate(self.sequences): # index on sequences
            for i in range(self.sizes[index] - 1): # index on line numbers (i.e. time_steps)
                images_paths.append([self.get_image_paths(sequence, i), self.get_image_paths(sequence, i + 1)]) # two consecutive image paths --> [num_sequences, num_lines, 2]
                pose1 = self.matrix_rt(self.poses[index][i]) # --> elements_in_the_line is passed into matrix_rt() to obtain 4x4 homogeneous pose matrix
                pose2 = self.matrix_rt(self.poses[index][i + 1])
                pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
                R = pose2wrt1[0:3, 0:3]
                t = pose2wrt1[0:3, 3]
                angles = rotationMatrixToEulerAngles(R)
                odometries.append(np.concatenate((t, angles))) # --> [ ,3(i.e. [x,y,z])]
        return np.array(images_paths), np.array(odometries)  # --> [num_squences*num_lines(total_time_steps), 2] , [num_squences*num_lines(total_time_steps), 6 (3 (i.e. (t, angles)) + 3(e.g. [x,y,z] or t) = 6)]
    


def main():
    path = "/home/cangozpi/Desktop/Docker_shared/Computer Vision for Autonomous Driving/MagicVO/dataset"
    train_dataset = VisualOdometryDataset(path, 192, 640, ['00'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    for i, batch in enumerate(train_dataloader):
        print("batch_size: ", len(batch),"\t\t batch_no: ", i)


if __name__ == "__main__":
    main()
