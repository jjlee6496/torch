import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Contracting Path
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        
        # Expanding Path
        self.dropout5 = nn.Dropout2d(p=0.5)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv9_3 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(2, 1, kernel_size=1)
    
    def forward(self, inputs):
        # Contracting Path
        conv1 = self.conv1(inputs)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        conv4 = self.conv4_2(conv4)
        drop4 = self.dropout4(conv4)
        pool4 = self.pool4(drop4)
        
        conv5 = self.conv5(pool4)
        conv5 = self.conv5_2(conv5)
        
        # Expanding Path
        drop5 = self.dropout5(conv5)
        up6 = self.up6(drop5)
        conv6 = self.conv6(torch.cat([drop4, up6], dim=1))
        conv6 = self.conv6_2(conv6)
        
        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], dim=1))
        conv7 = self.conv7_2(conv7)
        
        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], dim=1))
        conv8 = self.conv8_2(conv8)
        
        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], dim=1))
        conv9 = self.conv9_2(conv9)
        conv9 = self.conv9_3(conv9)
        conv10 = self.conv10(conv9)
        
        return conv10
