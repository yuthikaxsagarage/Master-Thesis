from os import pread
from this import d
from matplotlib import image
import torch

from imageio import imread, imsave
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from inverse_warp import pose_vec2mat
from scipy.ndimage.interpolation import zoom

from inverse_warp import *
from kitti_eval.kitti_odometry import umeyama_alignment

import models

from util import tensor2array
import cv2



class PoseEstimator():
    
    @torch.no_grad()
    def __init__(self, imagePath='./dataset/sequences/09/image_2/', preTrainedPoseNetwork='/home/anya/visual_odometry/visual_odometry_thesis/Thesis/checkpoints/resnet18_attention/07-31-19:55/exp_pose_model_best.pth.tar'):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.imagePath = imagePath
        self.groundTruthPath = './kitti_eval/gt_poses/09.txt'
        self.weights_pose = torch.load(preTrainedPoseNetwork)
        self.image_height = 256
        self.image_width = 832
        self.image_idx = 0
        self.pose_net = models.PoseAttentionNet().to(self.device)
        self.pose_net.load_state_dict(self.weights_pose['state_dict'], strict=False)
        self.pose_net.eval()
        self.trueX, self.trueY, self.trueZ, self.predX, self.predY, self.predZ = 0, 0, 0, 0 , 0 ,0
        self.image_dir = Path(imagePath)
        self.tensor_img2 = None
        test_files = sum([self.image_dir.files('*.{}'.format(ext)) for ext in ['png', 'jpg', 'bmp']], [])
        test_files.sort()
        self.test_files = test_files
        self.pos_xz = []
        self.final_poses = {}
        self.global_pose = np.eye(4)
        self.poses = [self.global_pose[0:3, :].reshape(1, 12)]
        self.gtPoses = self.load_poses_from_txt(self.groundTruthPath)
    
    def setInitialState(self):
        self.predX = 0
        self.predZ = 0
        self.trueX = 0
        self.trueZ = 0
        self.image_idx = 0
        self.pos_xz = []
        self.final_poses = {}
        self.global_pose = np.eye(4)
        self.poses = [self.global_pose[0:3, :].reshape(1, 12)]
    
    def setImagePath(self, path):
        self.imagePath = path
        self.image_dir = Path(path)
        test_files = sum([self.image_dir.files('*.{}'.format(ext)) for ext in ['png', 'jpg', 'bmp']], [])
        test_files.sort()
        self.test_files = test_files
        print(test_files)
    
    def setGroundTruthPath(self, path):
       self.groundTruthPath = path
       self.gtPoses = self.load_poses_from_txt(path)
        
    @torch.no_grad()
    def load_tensor_image(self, filename, img_width, img_height):
        img = imread(filename).astype(np.float32)
        h, w, _ = img.shape
        if (h != img_height or w != img_width):
            img = imresize(img, (img_height, img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(self.device)
        return tensor_img
      
    @torch.no_grad()
    def calculatePose(self, ref_img, current_image, image_idx):      
        self.image_idx = image_idx
        tensor_img1 = self.load_tensor_image(ref_img, self.image_width, self.image_height)
        tensor_img2 = self.load_tensor_image(current_image, self.image_width, self.image_height)
        self.pose = position = self.pose_net(tensor_img1, tensor_img2)

        self.pose_mat = pose_vec2mat(self.pose).squeeze(0).cpu().numpy()
     
        self.pose_mat = np.vstack([self.pose_mat, np.array([0, 0, 0, 1])])
        self.global_pose = self.global_pose @  np.linalg.inv(self.pose_mat)
        dof12pose = self.global_pose[0:3, :].reshape(1, 12).squeeze()
        self.poses.append(self.global_pose[0:3, :])
        self.final_poses[image_idx-1] = self.homogeneous_pose(dof12pose)
        
        scale = 1
        if(image_idx >1):
           scale = self.calculateScale()
    
        # print(scale)
        self.predX = dof12pose[3]*scale
        self.predZ = dof12pose[11]*scale      

   
        # print(self.gtPoses[image_idx]), homogeneous_pose
        self.trueX = self.gtPoses[image_idx][0][3]
        self.trueZ = self.gtPoses[image_idx][2][3]
        
        self.tensor_img1 = self.tensor_img2        


    def calculateScale(self) :
        poses_gt = self.gtPoses
        

        pred_0 = self.final_poses[0]
        gt_0 = poses_gt[0]
   
        adjustedFinalPoses = []
        

        for cnt in self.final_poses:
            adjustedFinalPoses.append(np.linalg.inv(pred_0) @ self.final_poses[cnt])
            poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]
        xyz_gt = []
        xyz_result = []
        
        for cnt in range(len(adjustedFinalPoses)):
            xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
            xyz_result.append([adjustedFinalPoses[cnt][0, 3], adjustedFinalPoses[cnt][1, 3], adjustedFinalPoses[cnt][2, 3]])
           
   
        xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
        xyz_result = np.asarray(xyz_result).transpose(1, 0)
            

        r, t, scale = umeyama_alignment(xyz_result, xyz_gt, True)
            
        return scale

    def load_poses_from_txt(self, file_name):
        """Load poses from txt (KITTI format)
            gets 12 numbers from the file 
            which is a 3x4 format
            
            computes homogenous poses 
        """
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ") if i != ""]
            withIdx = len(line_split) == 13
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses
    
    def homogeneous_pose(self, pose):
        i = 0
        P = np.eye(4)
        for row in range(3):
            for col in range(4):
                P[row, col] = pose[i]
                i=i+1
        return P