from ast import arg
from turtle import forward
import torch
import torch.nn.functional as F
import math
from torch.nn import init
import torch.nn as nn
import torch

class BYOT(torch.nn.Module):
    def __init__(self,channel):
        super().__init__()   

        self.byot1 = nn.Sequential(
            nn.Linear(channel, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 30),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(30, 2),
            nn.LeakyReLU(),
            nn.Dropout(0.4)
        )
    def forward(self,x, tem = 1):
        B = x.size(0)
        p1 = self.byot1(x.mean(1).view((B,-1)))
        return F.log_softmax(p1/tem, 1), F.softmax(p1/tem, 1)
import torch.nn as  nn
import numpy as np
class encoder_mlp(nn.Module):
    def __init__(self,in_channel, out_channel, mid_channel) -> None:
        super().__init__()
        mask = np.zeros((100,100))
        for i in range(100):
            for j in range(i+1, 100):
                mask[i,j]=1
        self.mask = torch.Tensor(mask)
        self.nn1 = nn.Linear(int(in_channel * (in_channel - 1) / 2), mid_channel)
        self.nn2 = nn.Linear(mid_channel, mid_channel * 2)
        self.nn3 = nn.Linear(mid_channel * 2, out_channel)

    def forward(self, x):
        device = x.device
        self.mask = self.mask.to(device)
        x = x[...,self.mask == 1]
        n = x.size(0)
        x = x.view(n,-1)
        x = F.dropout(F.leaky_relu(self.nn1(x)), 0.1)
        x = F.dropout(F.leaky_relu(self.nn2(x)), 0.1)
        x = F.dropout(F.leaky_relu(self.nn3(x)), 0.1)
        return x

class MGN(torch.nn.Module):
    def __init__(self, in_channel, kernel_size, num_classes=2,modal=1,args=None):
        super(MGN5, self).__init__()
        self.in_planes = 1 
        self.d = kernel_size
        self.modal = 1
        self.sink = 2
        self.channel = args.channel
        self.ll = args.layer
        self.ab = args.ab
        num_layers = args.gru
        if self.ab == 0:
            print("MLP")
            self.psi_1 = nn.Conv2d(1, self.channel, (kernel_size, 1), (kernel_size, 1))
            self.psi_2 = nn.Conv2d(1, self.channel, (kernel_size, 1), (kernel_size, 1))
        elif self.ab == 1:
            print("GRAPH")
            self.psi_1  = GraphConvolution(in_features=self.d,out_features=self.channel)
            self.psi_2  = GraphConvolution(in_features=self.d,out_features=self.channel)
        elif self.ab == 2:
            print("GRU")
            self.psi_1  = nn.GRU(input_size=self.d, hidden_size=args.channel, num_layers=num_layers)
            self.psi_2  = nn.GRU(input_size=self.d, hidden_size=args.channel, num_layers=num_layers)
        elif self.ab == 3:
            print("LSTM")
            self.psi_1  = nn.LSTM(input_size=self.d, hidden_size=args.channel, num_layers=num_layers)
            self.psi_2  = nn.LSTM(input_size=self.d, hidden_size=args.channel, num_layers=num_layers)

        self.mlp1 = nn.Sequential(
            nn.Linear(self.channel * self.sink, self.channel * self.sink * 2),
            nn.ReLU(),
            nn.Linear(self.channel * self.sink * 2, self.channel * self.sink),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.channel * self.sink, self.channel * self.sink * 2),
            nn.ReLU(),
            nn.Linear(self.channel * self.sink * 2, self.channel * self.sink),
        )
        self.dense1 = nn.Linear(self.channel * self.sink * self.modal ,128)
        self.dense2 = nn.Linear(128,30)
        self.dense3 = nn.Linear(30,num_classes)

        self.se = nn.Sequential(
            nn.Linear(self.d ,self.d //4),
            nn.ReLU(),
            nn.Linear(self.d //4,self.d),
            nn.Sigmoid()
        )
        self.byot1 = BYOT(self.channel)
        self.byot2 = BYOT(self.channel)

        self.ht_loss = 0.
        self.kd_loss = 0.
    def forward(self, x1,x2, tem =1):
        if x2 is None:
            x2 = x1
        elif x1 is None:
            x1 = x2
        # eac       
        if self.ab == 0:
            h_s = self.psi_1(x1).squeeze(2).permute(0,2,1)
            h_t = self.psi_2(x2).squeeze(2).permute(0,2,1)
        elif self.ab == 1:
            h_s = self.psi_1(x[:,0]).squeeze(2)#.mean(2)#.permute(0,2,1)
            h_t = self.psi_1(x[:,1]).squeeze(2)#.mean(2)#.permute(0,2,1)
        #lstm
        else:
            h_s = self.psi_1(x[:,0])[0].squeeze(2)#.permute(0,2,1)
            h_t = self.psi_1(x[:,1])[0].squeeze(2)#.permute(0,2,1)

        h_st1 = torch.cat((h_s, h_t), dim=2)

        x1 = self.mlp1(h_st1)
        B = x1.size(0)
        out = x1.mean(1).view((B,-1))
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)

        out_p = F.softmax(out,1)

        out1_log, out1_p = self.byot1(h_s)
        out2_log, out2_p = self.byot1(h_t)

        return out_p, out1_p, out2_p