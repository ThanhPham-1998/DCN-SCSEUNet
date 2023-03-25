from torchvision.ops.deform_conv import DeformConv2d
from torch.nn import functional as F
from torch import nn
import torch


def _init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, DeformConv2d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# SCSE module
class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, r)  # 16
        self.channel_gate = ChannelGate2d(in_ch)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2  # x = g1*x + g2*x
        return x


# Space Gating
class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()
        self.in_ch = in_ch
        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x
        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        x = input_x * x

        return x


# Channel Gating
class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = input_x * x

        return x


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dcn=False):
        super().__init__()
        self.dcn = dcn
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if self.dcn:
            self.offset1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1 * 2 * 3 * 3,
                kernel_size=(3, 3),
                padding=1
            )
            self.mask1 = nn.Conv2d(
                in_channels=in_channels,
                kernel_size=(3, 3),
                out_channels=1 * 3 * 3,
                padding=1
            )
            self.conv1 = DeformConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1
            )
            self.offset2 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=1 * 2 * 3 * 3,
                kernel_size=(3, 3),
                padding=1
            )
            self.mask2 = nn.Conv2d(
                in_channels=out_channels,
                kernel_size=(3, 3),
                out_channels=1 * 3 * 3,
                padding=1
            )
            self.conv2 = DeformConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1
            )
            self.conv2 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1
            )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x) if not self.dcn else self.conv1(x, self.offset1(x), self.mask1(x))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) if not self.dcn else self.conv2(out, self.offset2(out), self.mask2(out))
        out = self.bn2(out)
        out = self.relu(out)
        return out


class ExpansiveBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dcn=False, r=1):
        super().__init__()
        self.dcn = dcn
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            output_padding=1,
            dilation=1
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU()
        if self.dcn:
            self.offset1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1 * 2 * 3 * 3,
                kernel_size=(3, 3),
                padding=1
            )
            self.mask1 = nn.Conv2d(
                in_channels=in_channels,
                kernel_size=(3, 3),
                out_channels=1 * 3 * 3,
                padding=1
            )
            self.conv1 = DeformConv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=(3, 3),
                padding=1
            )
            self.offset2 = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=1 * 2 * 3 * 3,
                kernel_size=(3, 3),
                padding=1
            )
            self.mask2 = nn.Conv2d(
                in_channels=mid_channels,
                kernel_size=(3, 3),
                out_channels=1 * 3 * 3,
                padding=1
            )
            self.conv2 = DeformConv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                padding=1,
                kernel_size=(3, 3)
            )
            self.conv2 = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1
            )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.spa_cha_gate = SCSE(in_ch=out_channels, r=r)
    
    def forward(self, x, e):
        d = self.up(x)
        out = torch.cat([e, d], dim=1)
        out = self.conv1(out) if not self.dcn else self.conv1(out, self.offset1(out), self.mask1(out))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) if not self.dcn else self.conv2(out, self.offset2(out), self.mask2(out))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.spa_cha_gate(out)
        return out


# Output layer
def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(1, 1), in_channels=in_channels, out_channels=out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block



class SCSEUNet(nn.Module):
    def __init__(self, num_classes=2, activation = 'softmax', r=1, dcn=False) -> None:
        super().__init__()
        self.dcn = dcn
        self.activation = activation
        # Encode
        self.conv_encode1 = nn.Sequential(
            ContractingBlock(in_channels=3, out_channels=32, dcn=dcn),
            SCSE(32, r=r))
        self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = nn.Sequential(
            ContractingBlock(in_channels=32, out_channels=64, dcn=dcn),
            SCSE(64, r=r))
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = nn.Sequential(
            ContractingBlock(in_channels=64, out_channels=128, dcn=dcn),
            SCSE(128, r=r))
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode4 = nn.Sequential(
            ContractingBlock(in_channels=128, out_channels=256, dcn=dcn),
            SCSE(256, r=r))
        self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Bottleneck
        if dcn:
            self.bottleneck_offset1 = nn.Conv2d(
                in_channels=256,
                out_channels=1 * 2 * 3 * 3,
                padding=1,
                kernel_size=(3, 3)
            )
            self.bottleneck_mask1 = nn.Conv2d(
                in_channels=256,
                out_channels=1 * 3 * 3,
                kernel_size=(3, 3),
                padding=1
            )
            self.bottleneck_conv1 = DeformConv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1
            )
            self.bottleneck_offset2 = nn.Conv2d(
                in_channels=512,
                out_channels=1 * 2 * 3 * 3,
                padding=1,
                kernel_size=(3, 3)
            )
            self.bottleneck_mask2 = nn.Conv2d(
                in_channels=512,
                out_channels=1 * 3 * 3,
                kernel_size=(3, 3),
                padding=1
            )
            self.bottleneck_conv2 = DeformConv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1
            )
        else:
            self.bottleneck_conv1 = nn.Conv2d(
                kernel_size=3,
                in_channels=256,
                out_channels=512,
                padding=1
            )
            self.bottleneck_conv2 = nn.Conv2d(
                kernel_size=3,
                in_channels=512,
                out_channels=512,
                padding=1
            )
        
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(512)
        self.bottleneck_scse = SCSE(512, r=r)

    
        # Decode
        self.conv_decode4 = ExpansiveBlock(512, 256, 256, dcn=dcn, r=r)
        self.conv_decode3 = ExpansiveBlock(256, 128, 128, dcn=dcn, r=r)
        self.conv_decode2 = ExpansiveBlock(128, 64, 64, dcn=dcn, r=r)
        self.conv_decode1 = ExpansiveBlock(64, 32, 32, dcn=dcn, r=r)
        self.final_layer = final_block(32, num_classes)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_pool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)

        # Bottleneck
        # bottleneck = self.bottleneck(encode_pool4)
        bottleneck = self.bottleneck_conv1(encode_pool4) if not self.dcn else \
                     self.bottleneck_conv1(
                                            encode_pool4,
                                            self.bottleneck_offset1(encode_pool4),
                                            self.bottleneck_mask1(encode_pool4)
                                        )
        bottleneck = self.bn1(bottleneck)
        bottleneck = self.relu(bottleneck)
        bottleneck = self.bottleneck_conv2(bottleneck) if not self.dcn else \
                     self.bottleneck_conv2(
                                            bottleneck,
                                            self.bottleneck_offset2(bottleneck),
                                            self.bottleneck_mask2(bottleneck)
                                        )
        bottleneck = self.bn2(bottleneck)
        bottleneck = self.relu(bottleneck)
        bottleneck = self.bottleneck_scse(bottleneck) 

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)

        final_layer = self.final_layer(decode_block1)

        if self.activation == "softmax":
            out = final_layer.softmax(dim=1)  # Can be annotated, according to the situation
        elif self.activation == "sigmoid":
            out = final_layer.sigmoid()
        return out

if __name__ == "__main__":
    x = torch.rand(3, 3, 224, 224)
    model = SCSEUNet(r=16, dcn=True)
    _init_weights(module=model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    print(model(x).shape)
