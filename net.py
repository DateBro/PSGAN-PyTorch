import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.spectral_norm import spectral_norm as SpectralNorm


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, repeat_num=3, norm='SN'):
        super(Discriminator, self).__init__()

        layers = []
        if norm == 'SN':
            layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm == 'SN':
                layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        # k_size = int(image_size / np.power(2, repeat_num))
        if norm == 'SN':
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        if norm == 'SN':
            self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False)

        # conv1 remain the last square size, 256*256-->30*30
        # self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False))
        # conv2 output a single number

    def forward(self, x):
        h = self.main(x)
        out_makeup = self.conv1(h)
        return out_makeup.squeeze()


class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))

        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])

        return [out[key] for key in out_keys]


# Makeup Apply Network(MANet)
class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Generator, self).__init__()

        encoder_layers = []
        encoder_layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        # MANet设置没有affine
        encoder_layers.append(nn.InstanceNorm2d(conv_dim, affine=False))
        encoder_layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            encoder_layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            encoder_layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=False))
            encoder_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(3):
            encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        decoder_layers = []
        for i in range(3):
            decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            decoder_layers.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            decoder_layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            decoder_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        decoder_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        decoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.MDNet = MDNet()
        self.AMM = AMM()

    def forward(self, source_image, reference_image):
        fm_source = self.encoder(source_image)
        fm_reference = self.MDNet(reference_image)
        morphed_fm = self.AMM(fm_source, fm_reference)
        result = self.decoder(morphed_fm)
        return result


class MDNet(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    # MDNet is similar to the encoder of StarGAN
    def __init__(self, conv_dim=64, repeat_num=3):
        super(MDNet, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, reference_image):
        fm_reference = self.main(reference_image)
        return fm_reference


# AMM暂时先不用landmark detector，先试试一般的attention效果如何
# feature map也先不乘以visual_feature_weight
# 这里attention部分的计算也先存疑，因为论文中提到x和y需要是同一个区域，但结构图中是softmax
class AMM(nn.Module):
    """Attentive Makeup Morphing module"""

    def __init__(self):
        super(AMM, self).__init__()
        self.visual_feature_weight = 0.01
        self.lambda_matrix_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.beta_matrix_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fm_source, fm_reference):
        batch_size, channels, width, height = fm_reference.size()
        old_lambda_matrix = self.lambda_matrix_conv(fm_reference).view(batch_size, -1, width * height)
        old_beta_matrix = self.beta_matrix_conv(fm_reference).view(batch_size, -1, width * height)

        # reshape后fm的形状是C*(H*W)
        temp_fm_reference = fm_reference.view(batch_size, -1, height * width)
        # print('temp_fm_reference shape: ', temp_fm_reference.shape)
        # fm_source 在reshape后需要transpose成(H*W)*C
        temp_fm_source = fm_source.view(batch_size, -1, height * width).permute(0, 2, 1)
        # print('temp_fm_source shape: ', temp_fm_source.shape)
        # energy的形状应该是N*N，N=H*W
        energy = torch.bmm(temp_fm_source, temp_fm_reference)
        attention_map = self.softmax(energy)

        new_lambda_matrix = torch.bmm(old_lambda_matrix, attention_map.permute(0, 2, 1))
        new_beta_matrix = torch.bmm(old_beta_matrix, attention_map.permute(0, 2, 1))
        new_lambda_matrix = new_lambda_matrix.view(batch_size, 1, width, height)
        new_beta_matrix = new_beta_matrix.view(batch_size, 1, width, height)

        # 对feature_map_source进行修改
        lambda_tensor = new_lambda_matrix.expand(batch_size, 256, width, height)
        beta_tensor = new_beta_matrix.expand(batch_size, 256, width, height)
        morphed_fm_source = torch.mul(lambda_tensor, fm_source)
        morphed_fm_source = torch.add(morphed_fm_source, beta_tensor)

        return morphed_fm_source
