from argparse import ArgumentParser
import time
import numpy as np 
import cv2
import torch

import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt
kMinNumFeature = 1000
lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class ClassicalVisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0

        self.new_frame = None
        self.last_frame = None
        self.cam = cam
        self.cur_R = None
        self.focal = cam.fx
        self.px_ref = None
        self.px_cur = None
        self.pp = (cam.cx, cam.cy)
        self.detector = cv2.xfeatures2d.SIFT_create() 
      

    def featureTracking(self, image_ref, image_cur, px_ref):

        kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]
        pts1= kp1
        pts2 = kp2
        return pts1, pts2
    
    def processFrame(self):
        px_ref = self.detector.detect(self.new_frame)
        px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
        self.px_ref, self.px_cur = self.featureTracking(self.last_frame, self.new_frame, px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)       
        self.cur_R = R
        return R
        
    def update(self, ref_frame, tgt_frame):
        ref_frame = cv2.normalize(ref_frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        tgt_frame = cv2.normalize(tgt_frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        self.new_frame = tgt_frame
        self.last_frame = ref_frame
     
        return self.processFrame()
