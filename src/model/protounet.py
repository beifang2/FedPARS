import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class PSP_Module(nn.Module):
    def __init__(self, in_channels, out_channels, bin):
        super(PSP_Module, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(bin)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.size()
        x = self.global_avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size[2:], mode='bilinear', align_corners=True)
        return x


class PSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSP, self).__init__()
        bins = [1, 2, 3, 6]
        self.psp1 = PSP_Module(in_channels, out_channels, bins[0])
        self.psp2 = PSP_Module(in_channels, out_channels, bins[1])
        self.psp3 = PSP_Module(in_channels, out_channels, bins[2])
        self.psp4 = PSP_Module(in_channels, out_channels, bins[3])

    def forward(self, x):
        x1 = self.psp1(x)
        x2 = self.psp2(x)
        x3 = self.psp3(x)
        x4 = self.psp4(x)
        out = torch.cat([x, x1, x2, x3, x4], dim=1)
        return out

# def __init__(self, channel=3, filters=[64, 128, 256, 512, 1024]):
class ProtoUnet(nn.Module):
    def __init__(self, channel=3, filters=[32, 64, 128, 256, 512],num_prototype=3,gamma_list=[0.5,0.5]):
        super(ProtoUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.up_residual_conv5 = ResidualConv(filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            ResidualConv(2, 1, 1, 1),
            # nn.LogSoftmax(dim=1),
            nn.Sigmoid()
        )
        self.output_layer1 = nn.Sequential(
            ResidualConv(filters[0], 1, 1, 1),
            # nn.LogSoftmax(dim=1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

        # self.args = args_parser()

        in_channels = 32
        self.num_classes = 2
        self.num_prototype = num_prototype
        self.cls_head = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualConv(filters[0], filters[0], 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(0.10)
        )
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True)

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

        self.update_prototype = True
        self.gamma = gamma_list

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma[k], debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        # if dist.is_available() and dist.is_initialized():
        #     protos = self.prototypes.data.clone()
        #     dist.all_reduce(protos.div_(dist.get_world_size()))
        #     self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def forward(self, m, gt_semantic_seg=None, pretrain_prototype=False):
        # Encode1
        # stage1
        x1 = self.input_layer(m) + self.input_skip(m)  # 950*750

        # stage2
        x2 = self.residual_conv_1(x1)  # 475*375
        # stage3
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        x5 = self.bridge(x4)

        # Decode
        x5 = self.upsample_1(x5)  # 512*20*20

        x6 = torch.cat([x5, x4], dim=1)  # 768*20*20
        x7 = self.up_residual_conv1(x6)  # 256*20*20

        x7 = self.upsample_2(x7)  # 256*40*40

        x8 = torch.cat([x7, x3], dim=1)  # 384*40*40

        x9 = self.up_residual_conv2(x8)  # 128*40*40

        x9 = self.upsample_3(x9)  # 128*80*80
        x10 = torch.cat([x9, x2], dim=1)  # 192*80*80

        x11 = self.up_residual_conv3(x10)  # 64*80*80

        x11 = self.upsample_4(x11)  # 64*160*160
        x12 = torch.cat([x11, x1], dim=1)  # 96*160*160

        x13 = self.up_residual_conv4(x12)
        x14 = self.up_residual_conv5(x13)  # 2 32 512 512
        # if self.args.usepro == True:
        return prototypes(self, m, x14, gt_semantic_seg)

        # output = self.output_layer1(x14)
        # return output, x14


def prototypes(self, m, x, gt_semantic_seg=None):
    feats = x
    c = self.cls_head(feats)

    c = self.proj_head(c)
    _c = rearrange(c, 'b c h w -> (b h w) c')
    _c = self.feat_norm(_c)
    _c = l2_normalize(_c)

    self.prototypes.data.copy_(l2_normalize(self.prototypes))

    # n: h*w, k: num_class, m: num_prototype
    masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

    out_seg = torch.amax(masks, dim=1)
    out_seg = self.mask_norm(out_seg)
    out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])

    if gt_semantic_seg is None:
        pred_seg = out_seg
        return pred_seg, self.prototypes
    if gt_semantic_seg is not None:
        gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)
        contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
        return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}, self.prototypes

class ProjectionHead(nn.Module):
    def __init__(self, in_channels, proj_dim=256, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(in_channels, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t()  # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs


if __name__ == "__main__":
    model = ProtoUnet()
    a = torch.rand(1, 3, 256, 256)
    b,_ = model(a)
    print(b.shape)
