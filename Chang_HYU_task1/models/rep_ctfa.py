import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from Chang_HYU_task1.models.helpers.utils import make_divisible
from torchvision.ops.misc import Conv2dNormActivation
import math


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GRN(nn.Module):
    """Global Response Normalization (from ConvNeXt V2)"""

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # (B,C,H,W)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

    
class CTFAttention(nn.Module):
    """
    Channel-Time-Frequency Attention
    입력 : (B, C, F, T)  ← DCASE·ESC 전형적인 Mel-Spec 순서
    출력 : 동일 크기의 가중치로 스케일
    """
    def __init__(self, channels, gamma=2, b=1,
                 k_size_t=3, k_size_f=3, k_size_c=None):
        super().__init__()

        def _k(c, k):
            if k is not None:
                return k
            t = int(abs((math.log2(c) + b) / gamma))
            return t if t % 2 else t + 1

        k_t, k_f, k_c = (_k(channels, ks) for ks in (k_size_t, k_size_f, k_size_c))

        self.conv_t = nn.Conv1d(channels, channels, k_t,
                                padding=k_t // 2, groups=channels, bias=False)
        self.conv_f = nn.Conv1d(channels, channels, k_f,
                                padding=k_f // 2, groups=channels, bias=False)
        self.conv_c = nn.Conv1d(1, 1, k_c,
                                padding=k_c // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                 # x : (B, C, F, T)
        B, C, F, T = x.shape

        # ----- 채널 스코어 w_c -----
        w_c = x.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
        w_c = w_c.squeeze(-1).transpose(-1, -2)         # (B, 1, C)
        w_c = self.sigmoid(self.conv_c(w_c))            # (B, 1, C)
        w_c = w_c.transpose(-1, -2).unsqueeze(-1)       # (B, C, 1, 1)

        # ----- 시간 스코어 w_t -----
        w_t = x.mean(dim=2)                             # (B, C, T)
        w_t = self.sigmoid(self.conv_t(w_t))            # (B, C, T)
        w_t = w_t.unsqueeze(2)                          # (B, C, 1, T)  ← ★

        # ----- 주파수 스코어 w_f -----
        w_f = x.mean(dim=3)                             # (B, C, F)
        w_f = self.sigmoid(self.conv_f(w_f))            # (B, C, F)
        w_f = w_f.unsqueeze(3)                          # (B, C, F, 1)  ← ★

        # (F,1) ⊗ (1,T) ⇒ (F,T)  →  채널 스케일
        weight = w_c * (w_f * w_t)                      # (B, C, F, T)

        out = x * weight
        return out


class RepConv(nn.Module):
    """Depth-wise RepConv with multiple kernel shapes; re-parameterisable"""

    def __init__(self, in_channels, out_channels, stride=1, init_alphas=(1,1,1,1)):
        super().__init__()
        self.reparam_mode = False
        self.alpha = nn.Parameter(torch.tensor(init_alphas, dtype=torch.float32))

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 3), stride, (0, 1), groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv3x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 1), stride, (1, 0), groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.silu = nn.SiLU()
        self.fused_conv = None

    def forward(self, x):
        a = F.softplus(self.alpha)
        if self.reparam_mode and self.fused_conv is not None:
            return self.fused_conv(x)
        return self.silu(
            a[0]*self.conv3x3(x) +
            a[1]*self.conv1x1(x) +
            a[2]*self.conv1x3(x) +
            a[3]*self.conv3x1(x)
        )


class RepMobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate, stride):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        # 1×1 expand
        exp_conv = Conv2dNormActivation(
            in_channels, exp_channels, kernel_size=1, stride=1,
            norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU, inplace=False
        )

        # depth‑wise repconv
        depth_conv = RepConv(exp_channels, exp_channels, stride=stride)

        # 1×1 project
        proj_conv = Conv2dNormActivation(
            exp_channels, out_channels, kernel_size=1, stride=1,
            norm_layer=nn.BatchNorm2d, activation_layer=None, inplace=False
        )

        self.after_block_norm = GRN()
        self.after_block_activation = nn.SiLU()
        self.ctfa = CTFAttention(out_channels)

        # shortcut path
        if in_channels == out_channels:
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
                )
        else:
            self.use_shortcut = False

        self.block = nn.Sequential(exp_conv, depth_conv, proj_conv)

    def forward(self, x):
        if self.use_shortcut:
            x = self.block(x) + self.shortcut(x)
        else:
            x = self.block(x)
        x = self.ctfa(x)  # apply CTFA attention
        x = self.after_block_norm(x)
        x = self.after_block_activation(x)
        return x


class RepCTFA(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** s, 8)
                                                for s in range(n_stages)]
        self.total_block_count = 0

        # stem
        self.in_c = nn.Sequential(
            Conv2dNormActivation(in_channels, channels_per_stage[0] // 4,
                                 kernel_size=3, stride=2, inplace=False),
            Conv2dNormActivation(channels_per_stage[0] // 4, channels_per_stage[0],
                                 activation_layer=nn.SiLU, kernel_size=3, stride=2, inplace=False)
        )

        # stages
        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(
                channels_per_stage[stage_id], channels_per_stage[stage_id + 1],
                n_blocks[stage_id], strides=strides, expansion_rate=expansion_rate
            )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        # head
        self.feed_forward = nn.Sequential(
            nn.Conv2d(channels_per_stage[-1], n_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_classes),
            nn.AdaptiveAvgPool2d(1)
        )

        self.apply(initialize_weights)

    # ------------------------------------------------
    def _make_stage(self, in_c, out_c, n_blocks, strides, expansion_rate):
        stage = nn.Sequential()
        for _ in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f"b{block_id}"
            self.total_block_count += 1
            stride = strides.get(bname, (1, 1))
            stage.add_module(bname, RepMobileBlock(in_c, out_c, expansion_rate, stride))
            in_c = out_c
        return stage

    def _forward_conv(self, x):
        x = self.in_c(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        return x.squeeze(2).squeeze(2)


# ---------------- re‑parameterisation helpers ----------------

def fuse_conv_bn(conv, bn):
    if conv is None:
        return 0, 0
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(w.size(0), device=w.device)
    bn_w, bn_b = bn.weight, bn.bias
    std = torch.sqrt(bn.running_var + bn.eps)
    t = bn_w / std
    w_fused = w * t.reshape([-1, 1, 1, 1])
    b_fused = bn_b + (b - bn.running_mean) * t
    return w_fused, b_fused


def pad_tensor_to_3x3(kernel):
    h, w = kernel.size(2), kernel.size(3)
    pad_h, pad_w = 3 - h, 3 - w
    return F.pad(kernel, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])


def reparameterize_block(repconv: RepConv):
    a = F.softplus(repconv.alpha).detach().cpu()
    w3x3, b3x3 = fuse_conv_bn(repconv.conv3x3[0], repconv.conv3x3[1])
    w1x1, b1x1 = fuse_conv_bn(repconv.conv1x1[0], repconv.conv1x1[1])
    w1x3, b1x3 = fuse_conv_bn(repconv.conv1x3[0], repconv.conv1x3[1])
    w3x1, b3x1 = fuse_conv_bn(repconv.conv3x1[0], repconv.conv3x1[1])
    w3x3 *= a[0];   b3x3 *= a[0]
    w1x1 *= a[1];   b1x1 *= a[1]
    w1x3 *= a[2];   b1x3 *= a[2]
    w3x1 *= a[3];   b3x1 *= a[3]

    w1x1 = pad_tensor_to_3x3(w1x1)
    w1x3 = pad_tensor_to_3x3(w1x3)
    w3x1 = pad_tensor_to_3x3(w3x1)

    w_final = w3x3 + w1x1 + w1x3 + w3x1
    b_final = b3x3 + b1x1 + b1x3 + b3x1

    fused_conv = nn.Sequential(
        nn.Conv2d(
            in_channels=repconv.conv3x3[0].in_channels,
            out_channels=repconv.conv3x3[0].out_channels,
            kernel_size=3,
            stride=repconv.conv3x3[0].stride,
            padding=1,
            bias=True,
            groups=repconv.conv3x3[0].groups,
        ),
        repconv.silu,
    )
    fused_conv[0].weight.data = w_final
    fused_conv[0].bias.data = b_final

    repconv.fused_conv = fused_conv
    repconv.reparam_mode = True
    # delete branches to free memory
    del repconv.conv3x3, repconv.conv1x1, repconv.conv1x3, repconv.conv3x1


def reparameterize_model(model: nn.Module):
    model = copy.deepcopy(model)

    def convert(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, RepConv):
                reparameterize_block(child)
            else:
                convert(child)

    convert(model)
    return model


def get_model(n_classes: int = 10, in_channels: int = 1, base_channels: int = 32,
              channels_multiplier: float = 1.8, expansion_rate: float = 2.3,
              n_blocks=(3, 2, 1), strides=None):
    if strides is None:
        strides = dict(b2=(2, 2), b4=(2, 1))

    cfg = dict(
        n_classes=n_classes,
        in_channels=in_channels,
        base_channels=base_channels,
        channels_multiplier=channels_multiplier,
        expansion_rate=expansion_rate,
        n_blocks=n_blocks,
        strides=strides,
    )
    model = RepCTFA(cfg)
    return model