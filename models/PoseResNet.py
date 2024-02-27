# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from numpy import reshape

import torch
import torch.nn as nn
from collections import OrderedDict
from .resnet_encoder import *

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
    

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))
        
        self.query_fc = nn.Linear(208,208)
        self.key_fc   = nn.Linear(208,208)
        self.value_fc = nn.Linear(208,208)

        self.refine_convs = {}
        self.refine_convs[0] = nn.Conv2d(12* (num_frames_to_predict_for), 6 * (num_frames_to_predict_for),1,1,0)
        self.refine_convs[1] = nn.Conv2d(6 * (num_frames_to_predict_for), 6 * (num_frames_to_predict_for),3,1,1)
        self.refine_convs[2] = nn.Conv2d(6 * (num_frames_to_predict_for), 6 * (num_frames_to_predict_for),3,1,1)
        self.refine_convs[3] = nn.Conv2d(6 * (num_frames_to_predict_for), 6 * (num_frames_to_predict_for),3,1,1)

        self.refine_net = nn.ModuleList(list(self.refine_convs.values()))

        self.refine_pose_conv = nn.Conv2d(6 * (num_frames_to_predict_for), 6 * (num_frames_to_predict_for), 1)
        
    
    
    def atten_refine(self, inputs):
        B,C,H,W = inputs.size()
 
  
        inputs = inputs.view([B,C,H*W]) # B C N
   
     
        query = self.query_fc(inputs)
        key   = self.key_fc(inputs)
        value = self.value_fc(inputs)

        energy = torch.bmm(query, key.permute([0,2,1]))
        p_mat  = nn.functional.softmax(energy, 1)

        output = torch.bmm(p_mat, value)
        output = torch.cat([inputs, output], 1).view([B,2*C,H,W])

        for i in range(len(self.refine_convs)):
            output = self.refine_convs[i](output)
            output = self.relu(output)

        refine_output = self.refine_pose_conv(output)

        refine_output = refine_output.mean(3).mean(2)
        refine_output = 0.01 * refine_output.view(-1, self.num_frames_to_predict_for, 6) # B 2 1 6
        
        return refine_output
        

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        # last_features only has a 512*8*26 feature map
   
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        # features 6*8*26
        
     
        delta = self.atten_refine(out)
        delta = delta.squeeze()
        out = out.mean(3).mean(2)
       
       
        # reshape the tensor
        #  0.01 scaling from zhou17 observations that it led to stable convergence
        pose = 0.01 * out.view(-1, 6)
        #pose = 12 * 2 * 1 * 6
      
        return pose + delta


class PoseResNet(nn.Module):

    def __init__(self, num_layers = 18, pretrained = True):
        super(PoseResNet, self).__init__()
        self.encoder = ResnetEncoder(num_layers = num_layers, pretrained = pretrained, num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, img1, img2):
        x = torch.cat([img1,img2],1)
        features = self.encoder(x)
        pose = self.decoder([features])
        return pose

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = PoseResNet().cuda()
    model.train()

    tgt_img = torch.randn(4, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(4, 3, 256, 832).cuda() for i in range(2)]

    pose = model(tgt_img, ref_imgs[0])
