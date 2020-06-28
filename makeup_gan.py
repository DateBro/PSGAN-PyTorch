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

        encoder_layers = [nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                          nn.InstanceNorm2d(conv_dim, affine=False), nn.ReLU(inplace=True)]
        # MANet设置没有affine

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

    def forward(self, source_image, reference_image, mask_source, mask_ref, gamma=None, beta=None, ret=False,
                mode='train'):
        fm_source = self.encoder(source_image)
        fm_reference = self.MDNet(reference_image)
        if ret:
            gamma, beta = self.AMM(fm_source, fm_reference, mask_source, mask_ref, gamma=gamma, beta=beta, ret=ret,
                                   mode=mode)
            return [gamma, beta]
        morphed_fm = self.AMM(fm_source, fm_reference, mask_source, mask_ref, gamma=gamma, beta=beta, ret=ret,
                              mode=mode)
        result = self.decoder(morphed_fm)
        return result


class MDNet(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    # MDNet is similar to the encoder of StarGAN
    def __init__(self, conv_dim=64, repeat_num=3):
        super(MDNet, self).__init__()

        layers = [nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.InstanceNorm2d(conv_dim, affine=True), nn.ReLU(inplace=True)]

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


class AMM(nn.Module):
    """Attentive Makeup Morphing module"""

    def __init__(self):
        super(AMM, self).__init__()
        self.gamma_matrix_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.beta_matrix_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)

    def forward(self, fm_source, fm_reference, mask_source, mask_ref, gamma=None, beta=None, ret=False, mode='train'):
        old_gamma_matrix = self.gamma_matrix_conv(fm_reference)
        old_beta_matrix = self.beta_matrix_conv(fm_reference)

        old_gamma_matrix_source = self.gamma_matrix_conv(fm_source)
        old_beta_matrix_source = self.beta_matrix_conv(fm_source)

        if gamma is None:
            attention_map = self.raw_attention_map(fm_source, fm_reference)
            # gamma, beta = self.raw_atten_feature(mask_source, attention_map, old_gamma_matrix, old_beta_matrix,
            #                                      old_gamma_matrix_source, old_beta_matrix_source)
            gamma, beta = self.pure_atten_feature(attention_map, old_gamma_matrix, old_beta_matrix)
            if ret:
                return [gamma, beta]

        morphed_fm_source = fm_source * (1 + gamma) + beta

        return morphed_fm_source

    @staticmethod
    def raw_attention_map(fm_source, fm_reference):
        batch_size, channels, width, height = fm_reference.size()

        # reshape后fm的形状是C*(H*W)
        temp_fm_reference = fm_reference.view(batch_size, -1, height * width)
        # fm_source 在reshape后需要transpose成(H*W)*C
        temp_fm_source = fm_source.view(batch_size, -1, height * width).permute(0, 2, 1)
        # energy的形状应该是N*N，N=H*W
        energy = torch.bmm(temp_fm_source, temp_fm_reference)
        energy *= 200  # hyper parameters for visual feature
        attention_map = F.softmax(energy, dim=-1)

        return attention_map

    @staticmethod
    def raw_atten_feature(mask_source, attention_map, old_gamma_matrix, old_beta_matrix, old_gamma_matrix_source,
                          old_beta_matrix_source):
        batch_size, channels, width, height = old_gamma_matrix.size()
        old_gamma_matrix = old_gamma_matrix.view(batch_size, -1, width * height)
        old_beta_matrix = old_beta_matrix.view(batch_size, -1, width * height)
        new_gamma_matrix = torch.bmm(old_gamma_matrix, attention_map.permute(0, 2, 1))
        new_beta_matrix = torch.bmm(old_beta_matrix, attention_map.permute(0, 2, 1))
        new_gamma_matrix = new_gamma_matrix.view(-1, 1, width, height)
        new_beta_matrix = new_beta_matrix.view(-1, 1, width, height)

        reverse_mask_source = 1 - mask_source
        new_mask_source = F.interpolate(mask_source, size=new_gamma_matrix.shape[2:]).repeat(1, channels, 1, 1)
        new_reverse_mask_source = F.interpolate(reverse_mask_source, size=new_gamma_matrix.shape[2:]).repeat(1,
                                                                                                             channels,
                                                                                                             1, 1)
        gamma = new_gamma_matrix * new_mask_source + old_gamma_matrix_source * new_reverse_mask_source
        beta = new_beta_matrix * new_mask_source + old_beta_matrix_source * new_reverse_mask_source
        return gamma, beta

    # 只通过计算两个feature map的attention来修改gamma_matrix
    @staticmethod
    def pure_atten_feature(attention_map, old_gamma_matrix, old_beta_matrix):
        batch_size, channels, width, height = old_gamma_matrix.size()
        old_gamma_matrix = old_gamma_matrix.view(batch_size, -1, width * height)
        old_beta_matrix = old_beta_matrix.view(batch_size, -1, width * height)
        new_gamma_matrix = torch.bmm(old_gamma_matrix, attention_map.permute(0, 2, 1))
        new_beta_matrix = torch.bmm(old_beta_matrix, attention_map.permute(0, 2, 1))
        new_gamma_matrix = new_gamma_matrix.view(-1, 1, width, height)
        new_beta_matrix = new_beta_matrix.view(-1, 1, width, height)

        gamma = new_gamma_matrix
        beta = new_beta_matrix
        return gamma, beta

    # 下面是PSGAN中计算attention的方法，但需要的显存太大，而且感觉有一些不合理的地方
    @staticmethod
    def get_attention_map(mask_source, mask_ref, fm_source, fm_reference, mode='train'):
        HW = 64 * 64
        batch_size = 3

        # get 3 part fea using mask
        channels = fm_reference.shape[1]

        mask_source_re = F.interpolate(mask_source, size=64).repeat(1, channels, 1, 1)  # (3, c, h, w)
        fm_source = fm_source.repeat(3, 1, 1, 1)  # (3, c, h, w)
        # 计算 Attention 时 we only consider the pixels belonging to same facial region.
        fm_source = fm_source * mask_source_re  # (3, c, h, w) 3 stands for 3 parts

        mask_ref_re = F.interpolate(mask_ref, size=64).repeat(1, channels, 1, 1)
        fm_reference = fm_reference.repeat(3, 1, 1, 1)
        fm_reference = fm_reference * mask_ref_re

        theta_input = fm_source
        phi_input = fm_reference

        theta_target = theta_input.view(batch_size, -1, HW)
        theta_target = theta_target.permute(0, 2, 1)

        phi_source = phi_input.view(batch_size, -1, HW)

        weight = torch.bmm(theta_target, phi_source)  # (3, HW, HW)
        if mode == 'train':
            weight = weight.cpu()
            weight_ind = torch.LongTensor(weight.detach().numpy().nonzero())
            weight = weight.cuda()
            weight_ind = weight_ind.cuda()
        else:
            weight_ind = torch.LongTensor(weight.numpy().nonzero())
        weight *= 200  # hyper parameters for visual feature
        weight = F.softmax(weight, dim=-1)
        weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]
        return torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))

    @staticmethod
    def atten_feature(mask_ref, attention_map, old_gamma_matrix, old_beta_matrix):
        # 论文中有说gamma和beta的想法源于style transfer，但不是general style transfer，所以这里要用mask计算每个facial region的style
        batch_size, channels, width, height = old_gamma_matrix.size()

        mask_ref_re = F.interpolate(mask_ref, size=old_gamma_matrix.shape[2:]).repeat(1, channels, 1, 1)
        gamma_ref_re = old_gamma_matrix.repeat(3, 1, 1, 1)
        old_gamma_matrix = gamma_ref_re * mask_ref_re  # (3, c, h, w)
        beta_ref_re = old_beta_matrix.repeat(3, 1, 1, 1)
        old_beta_matrix = beta_ref_re * mask_ref_re

        old_gamma_matrix = old_gamma_matrix.view(3, 1, -1)
        old_beta_matrix = old_beta_matrix.view(3, 1, -1)

        old_gamma_matrix = old_gamma_matrix.permute(0, 2, 1)
        old_beta_matrix = old_beta_matrix.permute(0, 2, 1)
        new_gamma_matrix = torch.bmm(attention_map.to_dense(), old_gamma_matrix)
        new_beta_matrix = torch.bmm(attention_map.to_dense(), old_beta_matrix)
        gamma = new_gamma_matrix.view(-1, 1, width, height)  # (3, c, h, w)
        beta = new_beta_matrix.view(-1, 1, width, height)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)  # (c, h, w) combine the three parts
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta
