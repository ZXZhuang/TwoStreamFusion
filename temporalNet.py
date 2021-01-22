import torch.nn as nn
import torch.nn.functional as F

class TemporalNet(nn.Module):
    def __init__(self):
        super(TemporalNet, self).__init__()
        # input -> 2xLx224x224
        #卷积层1 7x7x96 卷积核大小7x7 步长2 输出通道数96
        self.conv1 = nn.Conv2d(18, 96, kernel_size=7, stride=2)
        #卷积层2 5x5x256 卷积核大小5x5 步长2 输出通道数256
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        #卷积层3 3x3x512 卷积核大小3x3 步长1 输出通道数512
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        #卷积层4 3x3x512 卷积核大小5x5 步长2 输出通道数512
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        #卷积层5 3x3x512 卷积核大小5x5 步长1 输出通道数512
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        #全连接层1
        self.fc1 = nn.Linear(512*2*2, 4096)
        #全连接层2
        self.fc2 = nn.Linear(4096, 2048)
        #classifier
        self.fc3 = nn.Linear(2048, 4)

    def forward(self, x):
        #batch_size = 256
        batch_size = x.size(0)
        #batch*3*224*224 -> batch*96*109*109
        out = self.conv1(x)
        #out = F.batch_norm(out)
        out = F.relu(out)
        #batch*96*109*109 -> batch*96*54*54
        out = F.max_pool2d(out, 3, 2)
        out = F.local_response_norm(out, 2)

        #batch*96*54*54 -> batch*256*25*25
        out = self.conv2(out)
        out = F.relu(out)
        #batch*256*25*25 -> batch*256*12*12
        out = F.max_pool2d(out, 3, 2)
        #out = F.local_response_norm(out, 2)

        #batch*256*12*12 -> batch*512*10*10
        out = self.conv3(out)
        out = F.relu(out)

        #batch*512*10*10 -> batch*512*8*8
        out = self.conv4(out)
        out = F.relu(out)

        #batch*512*8*8 -> batch*512*6*6
        out = self.conv5(out)
        out = F.relu(out)
        #batch*512*6*6 -> batch*512*2*2
        out = F.max_pool2d(out, 3, 2)

        #展平数据
        out = out.view(batch_size, -1)

        out = self.fc1(out)
        out = F.dropout(out)

        out = self.fc2(out)
        out = F.dropout(out)

        out = self.fc3(out)
        return out