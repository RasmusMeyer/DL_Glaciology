import torch
import torch.nn as nn

## The architecture of our ResUnet as described in the report.
## Generative AI was used as to create parts of this code under careful monitoring backed by litterature as described in code and report. 

class ResidualBlock(nn.Module):
    '''
    Our residual block consists of two convolutional layers including a shortcut connection to create a residual connection. 
    As described in He et al., (2016): https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        #3x3 but maintains dims if stride is 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        #Batch normalzation
        self.bn1 = nn.BatchNorm2d(out_channels)
        #Non linear ReLU activation function 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()

        #To be used in the encoder and decoder when the spatial dims do not match
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x) #identity copy
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity #and add onto output
        return self.relu(out)

class ResUNetEncoder(nn.Module):
    '''
    The encoder, bridge and decoder from the Unet architecture as described in Ronneberger et al. (2015) doi: 10.1007/978-3-319-24574-4_28
    and in the report.
    The encoder downsamples dims but increase the number of features. In ResUnet they are passed through ResidualBlocks between each downsampling step. 
    '''
    def __init__(self, input_channels, base_channels):
        super(ResUNetEncoder, self).__init__()

        #The initial conv layer with batch norm and Relu
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        #Res blocks with increasing feature depth and downsampling
        self.enc1 = ResidualBlock(base_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, stride=2) #Stride = 2 for downsampling
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = ResidualBlock(base_channels * 4, base_channels * 8, stride=2)

    #Encoder forward pass
    def forward(self, x): 
        x0 = self.initial(x) 
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        return x4, [x1, x2, x3]

class ResUNetBridge(nn.Module):
    '''
    The bridge connects the encoder and decoder in Unet. The data is passsed through resblock at the highest level of abstraction.
    '''
    def __init__(self, in_channels, out_channels):
        super(ResUNetBridge, self).__init__()
        self.bridge = ResidualBlock(in_channels, out_channels)
    
    def forward(self, x):
        return self.bridge(x)

class ResUNetDecoderBlock(nn.Module):
    '''
    This block implements the decoder part of the ResUNet architecture,
    upsamples the input feature map using transposed convolutions, 
    and then integrates the output with features from the encoder via skip connections. 
    The final result passes through a ResidualBlock to refine the output.
    '''
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResUNetDecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.resblock = ResidualBlock(mid_channels + out_channels, out_channels)
    
    def forward(self, x, skip_connection):
        x = self.upconv(x) #Upsamples 
        if x.size() != skip_connection.size():
            diffY = skip_connection.size(2) - x.size(2) #height diff
            diffX = skip_connection.size(3) - x.size(3) #Width diff
            x = nn.functional.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)) #Padding x to match dims and divides diff by two for equal padding
        
        x = torch.cat([x, skip_connection], dim=1)
        return self.resblock(x)

class ResUNetDecoder(nn.Module):
    '''
    Decoder in ResUNet. 
    This block upsamples the input feature maps and refines them 
    using skip connections from the encoder and residual blocks.
    '''
    def __init__(self, base_channels):
        super(ResUNetDecoder, self).__init__()
        self.dec4 = ResUNetDecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec3 = ResUNetDecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec2 = ResUNetDecoderBlock(base_channels * 2, base_channels, base_channels)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skips):
        x = self.dec4(x, skips[2])
        x = self.dec3(x, skips[1])
        x = self.dec2(x, skips[0])
        x = self.dec1(x)
        return x

class ResUNet(nn.Module):
    '''Implimentation of the Unet blocks, including the residual blocks from resnet inside the encoder, bridge and decoder'''
    def __init__(self, input_channels, output_channels, base_channels=64):
        super(ResUNet, self).__init__()
        self.encoder = ResUNetEncoder(input_channels, base_channels)
        self.bridge = ResUNetBridge(base_channels * 8, base_channels * 8)
        self.decoder = ResUNetDecoder(base_channels)
        self.final = nn.Conv2d(base_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bridge(x)
        x = self.decoder(x, skips)
        return self.final(x)