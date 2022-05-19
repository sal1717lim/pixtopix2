import torch
from torch import nn


class Upsample(nn.Module):
    def __init__(self, output_dim):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(output_dim, mode='bilinear')

    def forward(self, x):
        return self.upsample(x)
class EncoderCNN(nn.Module):

    def __init__(self):
        super(EncoderCNN, self).__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.vgg = nn.Sequential(
            *(vgg.features[i] for i in range(30)))

    def forward(self, images):
        return self.vgg(images)

class convBlock(nn.Module):
    def __init__(self, in_dim, out_dim, up):
        super(convBlock, self).__init__()

        #self.tconv = nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, output_padding=1 ,padding=1,padding_mode="zeros",kernel_size=(3, 3), stride=(2, 2))
        self.conv0 = nn.Conv2d(in_dim, out_dim, (3, 3), padding_mode="reflect")
        self.us = Upsample(output_dim=up)
        self.conv = nn.Conv2d(out_dim, out_dim, (3, 3), padding_mode="reflect")
        self.dp = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, skipped):
        x1 = self.conv0(x)
        x1 = self.us(x1)
        x1 = self.relu(x1)
        x1 = x1 + skipped
        x2 = self.conv(x1)
        x2 = self.relu(x2)
        x2 = self.dp(x2)
        x2 = self.bn(x2)
        return x2


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.vgg_rgb = EncoderCNN()
        self.vgg_depth = EncoderCNN()

        self.bn0 = nn.BatchNorm2d(512)
        self.convBlock1 = convBlock(512, 512, (32,32))
        self.convBlock2 = convBlock(512, 512, (32,32))
        self.convBlock3 = convBlock(512, 256, (64,64))
        self.convBlock4 = convBlock(256, 128, (128,128))
        self.convBlock5 = convBlock(128, 64, (256,256))
        self.output = nn.Conv2d(64,3,(1,1),padding=1)
        self.th = nn.Tanh()
    def forward(self, rgb, depth):
        Rgb = []
        Depth = []
        for i in range(len(self.vgg_rgb.vgg)):
            rgb = self.vgg_rgb.vgg[i](rgb)
            depth= self.vgg_depth.vgg[i](depth)
            if i in [3,8,15,22]:
                Rgb.append(rgb.clone())
                Depth.append(depth.clone())
        #deconvs incoming
        x = rgb + depth
        x = self.bn0(x)
        x = self.convBlock1(x, Rgb[-1] + Depth[-1])
        x = self.convBlock2(x, Rgb[-1] + Depth[-1])
        x = self.convBlock3(x, Rgb[-2] + Depth[-2])
        x = self.convBlock4(x, Rgb[-3] + Depth[-3])
        x = self.convBlock5(x, Rgb[-4] + Depth[-4])
        x = self.output(x)

        return self.th(x)



def test():
    x = torch.randn((1, 3, 256, 256))
    x1 = torch.randn((1, 3, 256, 256))
    model = Generator()
    preds = model(x,x1)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()