from os import pread
from this import d
import time
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



class DepthEstimator():
    
    @torch.no_grad()
    def __init__(self, imagePath='./dataset/sequences/00/image_2/', preTrainedDepthNetwork='/home/anya/visual_odometry/visual_odometry_thesis/Thesis/models/dispnet_model_best.pth.tar'):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.imagePath = imagePath
        self.weights_pose = torch.load(preTrainedDepthNetwork)
        self.image_height = 256
        self.image_width = 832
        self.disp_net = models.DispResNet(18, False).to(self.device)
        self.disp_net.load_state_dict(self.weights_pose['state_dict'], strict=False)
        self.disp_net.eval()       
         
    def setImagePath(self, path):
        self.imagePath = path
        self.image_dir = Path(path)
        test_files = sum([self.image_dir.files('*.{}'.format(ext)) for ext in ['png', 'jpg', 'bmp']], [])
        test_files.sort()
        self.test_files = test_files     
        
    @torch.no_grad()
    def load_tensor_image(self, img):
        h,w,_ = img.shape
        if (h != self.image_height or w != self.image_width):
            img = imresize(img, (self.image_height, self.image_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(self.device)
        return tensor_img

 
    @torch.no_grad()
    def calculateDepth(self, img):      
  
        tgt_img = self.load_tensor_image(img)
        output = self.disp_net(tgt_img)[0]
        disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
        img = disp[1:, :, :]
        img = np.transpose(disp, (1, 2, 0)).copy()
     
        return img
       
