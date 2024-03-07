import math
from torch_scatter import scatter_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageConvLayer(nn.Module):
    '''
    Image Convolution Layer
    '''

    def __init__(self, in_ch: int, out_ch: int, kernel_size=5) -> torch.Tensor:
        super(ImageConvLayer, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        x = self.point_conv(self.bn(x))
        x = self.act1(x)
        x = self.depth_conv(x)
        out = self.act2(x)

        return out


class GAFF(nn.Module):
    '''
    graph global-local contextual attention feature fusion module (GAFF)
    '''

    def __init__(self, channels: int = 64, r: int = 4) -> torch.Tensor:
        super(GAFF, self).__init__()
        inter_channels = int(channels * r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels,
                      kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels,
                      kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x.shape = [B,C,H,W] = y.shape
        xy = x + y
        loc_attn = self.local_att(xy)
        glo_attn = self.global_att(xy)
        lg_attn = loc_attn + glo_attn
        attn = self.sigmoid(lg_attn)

        out = x * attn + y * (1 - attn)

        return out


class RGCMF(nn.Module):
    """
    residual graph convolution modulate fusion layer (RGCMF)
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 adjacency_matrix: torch.Tensor) -> torch.Tensor:
        super(RGCMF, self).__init__()

        self.gaff = GAFF(channels=output_dim, r=2)

        self.bn = nn.BatchNorm1d(input_dim)
        self.act = nn.LeakyReLU(inplace=True)

        self.gcn_lin = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 64))

        self.fc = nn.Linear(input_dim, output_dim)
        # this dict for determining height and width of x(y), and then perform gaff!
        # default: h=w= math.sqrt(num_superpixel), i.e., h*w=num_superpixel!
        self.hw = {'8192': (128, 64),
                   '2048': (64, 32),
                   '512': (32, 16),
                   '128': (16, 8),
                   '32': (8, 4),
                   '8': (4, 2)}

        self.alpha = nn.Parameter(torch.ones(1))
        self.I = torch.eye(adjacency_matrix.shape[0], adjacency_matrix.shape[0],
                           requires_grad=False, device=device, dtype=torch.float32)
        self.mask = torch.ceil(adjacency_matrix * 0.00001)

    def forward(self, H):
        H = self.act(self.bn(H))
        # start graph conv
        HW_att = self.proj(H)  #
        XW = self.gcn_lin(H)  # XW in equ.9

        A_att = torch.sigmoid(torch.matmul(HW_att, HW_att.t()))
        zero_vec = -9e15 * torch.ones_like(A_att)
        A_hat = torch.where(self.mask > 0, A_att,
                            zero_vec) + self.alpha * self.I
        A_hat = F.softmax(A_hat, dim=1)
        X = torch.mm(A_hat, XW)
        # end graph conv

        # residual branch.
        Y = self.fc(H)

        # start gaff
        # reshape X from (N,C) to (B=1, C, H, W)
        X = X.transpose(0, 1)
        c, hw = X.shape
        # default: h=w=math.sqrt(hw)
        h = w = int(math.sqrt(hw))
        if self.hw.get(str(hw), None) is not None:
            h, w = self.hw[str(hw)]
        X = X.reshape(-1, h, w)
        X = X.unsqueeze(0)

        # reshape Y from (N,C) to (B=1, C, H, W)
        Y = Y.transpose(0, 1)
        Y = Y.reshape(-1, h, w)
        Y = Y.unsqueeze(0)

        out = self.gaff(X, Y)
        # reshape out from (B=1, C, H, W) to (N,C)
        out = out.squeeze().reshape(-1, hw).transpose(0, 1)
        # end gaff

        return out


class DPGUNet(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 channel: int,
                 class_count: int,
                 association_matrices: torch.Tensor,
                 adjacency_matrices: torch.Tensor) -> torch.Tensor:
        super(DPGUNet, self).__init__()
        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width
        self.association_matrices = association_matrices
        self.cmam_list = []  # cross-scale merging association matrix list
        self.alpha = nn.Parameter(torch.ones(1))

        for i in range(len(self.association_matrices)):
            if i == 0:
                self.cmam_list.append(self.association_matrices[i])
            else:
                self.cmam_list.append(
                    torch.mm(self.cmam_list[i-1], self.association_matrices[i]))

        self.adjacency_matrices = adjacency_matrices
        self.net_depth = len(association_matrices)
        self.association_matrices_t = []  # for pooling
        for i in range(len(association_matrices)):
            temp = association_matrices[i]
            self.association_matrices_t.append(
                (temp / (torch.sum(temp, dim=0, keepdim=True, dtype=torch.float32))).t())

        # layer_channels = 128
        dim_per_layer = [8, 16, 32, 64, 128]

        self.en_image_conv_layer = ImageConvLayer(
            self.channel, dim_per_layer[0], kernel_size=5)

        self.en_rgcmfs = nn.Sequential()
        for i in range(self.net_depth):
            self.en_rgcmfs.add_module('encoder_RGCMFs_' + str(i),
                                      RGCMF(dim_per_layer[i], dim_per_layer[i + 1],
                                            self.adjacency_matrices[i]))

        self.de_rgcmfs = nn.Sequential()
        for i in range(self.net_depth - 1):  # 4, 3, 2, 三层decoder
            self.de_rgcmfs.add_module('decoder_RGCMFs_' + str(i),
                                      RGCMF(dim_per_layer[-i - 1] + dim_per_layer[-i - 2],
                                            dim_per_layer[-i - 2],
                                            self.adjacency_matrices[-i - 2]))

        self.GPFL = nn.Sequential()
        for i in range(self.net_depth):
            self.GPFL.add_module('GPFL'+str(i),
                                 nn.Linear(dim_per_layer[-i-1], self.class_count))

        self.de_image_conv_layer = ImageConvLayer(
            dim_per_layer[0] + dim_per_layer[1], dim_per_layer[0], kernel_size=5)

        self.fc = nn.Linear(dim_per_layer[0], self.class_count)
        self.final_fc = nn.Linear(self.class_count, self.class_count)
        self.Softmax = nn.Softmax(-1)

    def forward(self, x: torch.Tensor):
        h, w, c = x.shape
        # [h, w, c]->[b=1, c, h, w]
        x = torch.unsqueeze(x.permute([2, 0, 1]), 0)

        # start encoder
        # image convolution layer in encoder
        x = self.en_image_conv_layer(x)
        # permutation
        # [b=1,c,h,w,] -> [h,w,c]
        x = torch.squeeze(x, 0).permute([1, 2, 0])
        # [h,w,c]->[n,c]
        H = x.reshape([h * w, -1])  # H.shape=[21025, 128]
        encoder_features = []
        decoder_features = []
        encoder_features.append(H)

        for i in range(len(self.en_rgcmfs)):

            # NGNAP
            row, col = torch.where(self.association_matrices[i])
            H_softmax_row = scatter_softmax(H, index=col, dim=0)  # row
            H_softmax_col = F.softmax(H, dim=-1)  # col
            H = self.alpha * \
                torch.mul(H, H_softmax_row) + (1 - self.alpha) * \
                torch.mul(H, H_softmax_col)
            # simple avg pooling
            H = torch.mm(self.association_matrices_t[i], H)

            # RGCMF
            H = self.en_rgcmfs[i](H)
            encoder_features.append(H)
        # end encoder

        # start decoder
        decoder_features.append(H)
        for i in range(len(self.de_rgcmfs)):
            # upsampling
            H = torch.mm(self.association_matrices[-i - 1], H)
            H = torch.cat([H, encoder_features[-i - 2]], dim=-1)
            # RGCMF
            H = self.de_rgcmfs[i](H)
            decoder_features.append(H)
        # upsampling
        H = torch.mm(self.association_matrices[0], H)
        H = torch.cat([H, encoder_features[0]], dim=-1)

        H = H.reshape([1, h, w, -1]).permute([0, 3, 1, 2])
        # image convolution layer in decoder
        final_features = self.de_image_conv_layer(H)
        # end decoder

        # start GPFL
        final_features = torch.squeeze(final_features, 0).permute([
            1, 2, 0]).reshape([h * w, -1])
        final_features = self.fc(final_features)

        for i in range(len(decoder_features)):
            # tmp.shape=[256*256, 8(16,32,64)]
            tmp = torch.mm(self.cmam_list[-1 - i], decoder_features[i])
            feat = self.GPFL[i](tmp)  # feat.shape=[256*256, 6]
            final_features += feat
        # end GPFL

        # fc and softmax
        final_features = self.final_fc(final_features)
        output = self.Softmax(final_features)

        return output
