import torch
import torch.nn as nn
import torch.nn.functional as F
from spatialNet import SpatialNet
from temporalNet import TemporalNet

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.spatialNet = SpatialNet()
        self.temporalNet = TemporalNet()
        self.fc = nn.Linear(8, 4)


    def forward(self, x, y):
        out_s = self.spatialNet(x)
        out_t = self.temporalNet(y)
        out = torch.cat((out_s, out_t), 1)
        out = self.fc(out)
        out = F.softmax(out)
        return out