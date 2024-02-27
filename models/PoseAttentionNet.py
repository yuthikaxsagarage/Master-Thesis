from __future__ import absolute_import, division, print_function
from numpy import reshape
import torchvision
import torch
import torch.nn as nn
from collections import OrderedDict
from .resnet_encoder import *
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T

class PoseCNN(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseCNN, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.depth = 0

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

        self.attention_convs = {}
        self.attention_convs[0] = nn.Conv2d(12* (num_frames_to_predict_for), 6 * (num_frames_to_predict_for),1,1,0)
        self.attention_convs[1] = nn.Conv2d(6 * (num_frames_to_predict_for), 6 * (num_frames_to_predict_for),3,1,1)
        self.attention_convs[2] = nn.Conv2d(6 * (num_frames_to_predict_for), 6 * (num_frames_to_predict_for),3,1,1)
        self.attention_convs[3] = nn.Conv2d(6 * (num_frames_to_predict_for), 6 * (num_frames_to_predict_for),3,1,1)

        self.attention_net = nn.ModuleList(list(self.attention_convs.values()))

        self.attention_pose_conv = nn.Conv2d(6 * (num_frames_to_predict_for), 6 * (num_frames_to_predict_for), 1)
        
    
    
    def attention_for_pose(self, inputs):
        B,C,H,W = inputs.size()
 
        inputs = inputs.view([B,C,H*W]) # B C N   
     
        query = self.query_fc(inputs)
        key   = self.key_fc(inputs)
        value = self.value_fc(inputs)

        energy = torch.bmm(query, key.permute([0,2,1]))
        p_mat  = nn.functional.softmax(energy, 1)

        output = torch.bmm(p_mat, value)
        output = torch.cat([inputs, output], 1).view([B,2*C,H,W])

        for i in range(len(self.attention_convs)):
            output = self.attention_convs[i](output)
            output = self.relu(output)

        attention_output = self.attention_pose_conv(output)

        attention_output = attention_output.mean(3).mean(2)
        attention_output = 0.01 * attention_output.view(-1, self.num_frames_to_predict_for, 6) # B 2 1 6
        
        return attention_output
        

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        # just before last_features has a 512*8*26 feature map
   
        output_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        output_features = torch.cat(output_features, 1)

        for i in range(3):
            output_features = self.convs[("pose", i)](output_features)
            if i != 2:
                output_features = self.relu(output_features)
     
        pose_attention = self.attention_for_pose(output_features)
        pose_attention = pose_attention.squeeze()
        output_features = output_features.mean(3).mean(2)
      
        # reshape the tensor
        #  0.01 scaling from zhou17 observations that it led to stable convergence
        pose = 0.01 * output_features.view(-1, 6)
        #pose = 4 * 6
 
        return pose + pose_attention


class PoseAttentionNet(nn.Module):

    def __init__(self, num_layers = 18, pretrained = True):
        super(PoseAttentionNet, self).__init__()
        self.encoder = ResnetEncoder(num_layers = num_layers,
                                     pretrained = pretrained, num_input_images=2)
        self.decoder = PoseCNN(self.encoder.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, img1, img2):
     
        x = torch.cat([img1,img2],1)
        features = self.encoder(x)
        pose = self.decoder([features])
        return pose
 
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = PoseAttentionNet().cuda()
    model.train()

    tgt_img = torch.randn(4, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(4, 3, 256, 832).cuda() for i in range(2)]

    pose = model(tgt_img, ref_imgs[0])
