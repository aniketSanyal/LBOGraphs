"""
Convolutional Neural Operator (CNO), 1D.

Reference implementation, kept faithful to the official code released with

    Raonic, Molinaro, De Ryck, Rohner, Bartolucci, Alaifari, Mishra,
    de Bezenac, "Convolutional Neural Operators for robust and accurate
    learning of PDEs", NeurIPS 2023 (arXiv:2302.01178)

    https://github.com/camlab-ethz/ConvolutionalNeuralOperator
    (CNO1d_simplified/CNO1d.py)

The CNO_LReLu / CNOBlock / LiftProjectBlock / ResidualBlock / ResNet / CNO1d
classes below reproduce the reference architecture: activations are applied
in the "operator" way (bandlimited up-sampling to 2x, LeakyReLU, then
down-sampling back to the target size, via antialiased bicubic
interpolation), all convolutions are 3x1 with padding 1, and the U-Net
style encoder/decoder carries invariant ResNet blocks on every level plus a
bottleneck ResNet, with ED_expansion blocks resampling the skip connections.

CNOOperator at the bottom is the only addition: a thin wrapper that adapts
CNO1d to the initial-condition -> final-state mapping produced by
equations.py, so it can be benchmarked against other architectures on the
same data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNO_LReLu(nn.Module):
    """
    Operator-consistent activation: up-sample to twice the input
    resolution, apply LeakyReLU there, then down-sample to out_size.
    Antialiased bicubic interpolation acts as the bandlimiting filter.
    """

    def __init__(self, in_size, out_size):
        super(CNO_LReLu, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(
            x.unsqueeze(2),
            size=(1, 2 * self.in_size),
            mode="bicubic",
            antialias=True,
        )
        x = self.act(x)
        x = F.interpolate(
            x,
            size=(1, self.out_size),
            mode="bicubic",
            antialias=True,
        )
        return x[:, :, 0]


class CNOBlock(nn.Module):
    """Convolution -> (BatchNorm) -> operator activation (in_size -> out_size)."""

    def __init__(self, in_channels, out_channels, in_size, out_size, use_bn=True):
        super(CNOBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size

        self.convolution = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )

        if use_bn:
            self.batch_norm = nn.BatchNorm1d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()

        self.act = CNO_LReLu(in_size=self.in_size, out_size=self.out_size)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)


class LiftProjectBlock(nn.Module):
    """Lifting / projection: a resolution-preserving CNOBlock then a conv."""

    def __init__(self, in_channels, out_channels, size, latent_dim=64):
        super(LiftProjectBlock, self).__init__()

        self.inter_CNOBlock = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=size,
            out_size=size,
            use_bn=False,
        )

        self.convolution = torch.nn.Conv1d(
            in_channels=latent_dim,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x


class ResidualBlock(nn.Module):
    """Resolution-invariant residual block (size in == size out)."""

    def __init__(self, channels, size, use_bn=True):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.size = size

        self.convolution1 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
        )
        self.convolution2 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
        )

        if use_bn:
            self.batch_norm1 = nn.BatchNorm1d(self.channels)
            self.batch_norm2 = nn.BatchNorm1d(self.channels)
        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        self.act = CNO_LReLu(in_size=self.size, out_size=self.size)

    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out


class ResNet(nn.Module):
    """Stack of num_blocks invariant ResidualBlocks."""

    def __init__(self, channels, size, num_blocks, use_bn=True):
        super(ResNet, self).__init__()
        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(
                ResidualBlock(channels=channels, size=size, use_bn=use_bn)
            )

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
        return x


class CNO1d(nn.Module):
    """
    1D Convolutional Neural Operator (reference architecture).

    in_dim/out_dim      : input/output channels (function components)
    size                : spatial resolution of the input grid
    N_layers            : number of down/up-sampling levels
    N_res               : invariant residual blocks per encoder level
    N_res_neck          : invariant residual blocks in the bottleneck
    channel_multiplier  : base channel width
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        size,
        N_layers,
        N_res=4,
        N_res_neck=4,
        channel_multiplier=16,
        use_bn=True,
    ):
        super(CNO1d, self).__init__()

        self.N_layers = int(N_layers)
        self.lift_dim = channel_multiplier // 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_multiplier = channel_multiplier

        # ---- channel widths -------------------------------------------------
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2**i * self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        # decoder levels (past the first) receive concatenated skips
        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]

        # ---- resolutions ----------------------------------------------------
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(size // 2**i)
            self.decoder_sizes.append(size // 2 ** (self.N_layers - i))

        # ---- lifting / projection ------------------------------------------
        self.lift = LiftProjectBlock(
            in_channels=in_dim,
            out_channels=self.encoder_features[0],
            size=size,
        )

        self.project = LiftProjectBlock(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=out_dim,
            size=size,
        )

        # ---- encoder / decoder ----------------------------------------------
        self.encoder = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i + 1],
                    in_size=self.encoder_sizes[i],
                    out_size=self.encoder_sizes[i + 1],
                    use_bn=use_bn,
                )
                for i in range(self.N_layers)
            ]
        )

        # resample skip connections to the matching decoder resolution
        self.ED_expansion = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i],
                    in_size=self.encoder_sizes[i],
                    out_size=self.decoder_sizes[self.N_layers - i],
                    use_bn=use_bn,
                )
                for i in range(self.N_layers + 1)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.decoder_features_in[i],
                    out_channels=self.decoder_features_out[i],
                    in_size=self.decoder_sizes[i],
                    out_size=self.decoder_sizes[i + 1],
                    use_bn=use_bn,
                )
                for i in range(self.N_layers)
            ]
        )

        # ---- invariant residual stacks --------------------------------------
        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        for l in range(self.N_layers):
            self.res_nets.append(
                ResNet(
                    channels=self.encoder_features[l],
                    size=self.encoder_sizes[l],
                    num_blocks=self.N_res,
                    use_bn=use_bn,
                )
            )

        self.res_net_neck = ResNet(
            channels=self.encoder_features[self.N_layers],
            size=self.encoder_sizes[self.N_layers],
            num_blocks=self.N_res_neck,
            use_bn=use_bn,
        )

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        x = self.lift(x)
        skip = []

        for i in range(self.N_layers):
            y = self.res_nets[i](x)
            skip.append(y)
            x = self.encoder[i](x)

        x = self.res_net_neck(x)

        for i in range(self.N_layers):
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)
            else:
                x = torch.cat(
                    (x, self.ED_expansion[self.N_layers - i](skip[-i])), 1
                )
            x = self.decoder[i](x)

        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)
        x = self.project(x)

        return x


# ---------------------------------------------------------------------------
# Benchmarking wrapper
#
# Not part of the reference CNO: this adapts CNO1d to the state-evolution
# task defined by equations.py, mapping an initial field u(0) on a domain to
# the final field u(T), so CNO can be compared against other architectures on
# identical data.
# ---------------------------------------------------------------------------


class CNOOperator(nn.Module):
    """
    Maps an initial state to a final state with CNO1d.

    Expects inputs shaped [batch, n_points, in_channels] (the layout produced
    by equations.py solutions) and returns [batch, n_points, out_channels].
    Internally transposes to the [batch, channels, n_points] layout CNO1d
    uses, so the channels-last convention of the rest of the project is kept
    at the interface.

    n_points is the number of spatial points in the discretization; for a
    GridDomain this is domain.n_points (interior points only).
    """

    def __init__(
        self,
        n_points,
        in_channels=1,
        out_channels=1,
        N_layers=3,
        N_res=4,
        N_res_neck=4,
        channel_multiplier=16,
        use_bn=True,
    ):
        super(CNOOperator, self).__init__()
        self.n_points = n_points

        self.cno = CNO1d(
            in_dim=in_channels,
            out_dim=out_channels,
            size=n_points,
            N_layers=N_layers,
            N_res=N_res,
            N_res_neck=N_res_neck,
            channel_multiplier=channel_multiplier,
            use_bn=use_bn,
        )

    def forward(self, u0):
        # [B, N, C] -> [B, C, N]
        x = u0.permute(0, 2, 1)
        x = self.cno(x)
        # [B, C, N] -> [B, N, C]
        return x.permute(0, 2, 1)
