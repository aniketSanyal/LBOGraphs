"""
Fourier Neural Operator (FNO), 1D.

Reference implementation, kept faithful to the official code released with

    Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar,
    "Fourier Neural Operator for Parametric Partial Differential
    Equations", ICLR 2021 (arXiv:2010.08895)

    https://github.com/neuraloperator/neuraloperator  (fourier_1d.py)

The SpectralConv1d and FNO1d classes below reproduce the reference
architecture: each Fourier layer truncates the signal to the lowest
`modes1` Fourier coefficients, multiplies them by a learned complex
weight tensor, and transforms back, run in parallel with a pointwise
(1x1) convolution acting as the local/bias path. Four such layers are
stacked with GELU between them (no activation after the last), wrapped
by a lifting Linear and a two-layer pointwise decoder. Following the
reference, the input coordinate grid is concatenated as an extra channel
before lifting. Reference defaults: modes=16, width=64.

FNOOperator at the bottom is the only addition: a thin wrapper that adapts
FNO1d to the initial-condition -> final-state mapping produced by
equations.py, so it can be benchmarked against other architectures on the
same data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: FFT -> keep the lowest modes1 coefficients and
    multiply them by learned complex weights -> inverse FFT.
    """

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of low-frequency Fourier modes kept (at most floor(N/2) + 1).
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, dtype=torch.cfloat
            )
        )

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator (reference architecture).

    modes : number of Fourier modes retained per spectral layer
    width : lifted channel width

    Input is [batch, x, in_channels]; the normalized coordinate grid is
    concatenated as an extra channel, so the lifting layer sees
    in_channels + 1. Output is [batch, x, out_channels].
    """

    def __init__(self, modes, width, in_channels=1, out_channels=1):
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lift: input channels + 1 coordinate channel -> width
        self.fc0 = nn.Linear(in_channels + 1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(
            np.linspace(0, 1, size_x), dtype=torch.float
        )
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# ---------------------------------------------------------------------------
# Benchmarking wrapper
#
# Not part of the reference FNO: this adapts FNO1d to the state-evolution
# task defined by equations.py, mapping an initial field u(0) on a domain to
# the final field u(T), so FNO can be compared against other architectures on
# identical data. Mirrors ref_models/cno.py's CNOOperator interface.
# ---------------------------------------------------------------------------


class FNOOperator(nn.Module):
    """
    Maps an initial state to a final state with FNO1d.

    Expects inputs shaped [batch, n_points, in_channels] (the layout
    produced by equations.py solutions) and returns
    [batch, n_points, out_channels] - FNO1d already uses this channels-last
    convention, so no transposition is needed at the interface.

    n_points is the number of spatial points in the discretization; for a
    GridDomain this is domain.n_points (interior points only). It is used
    only to clamp `modes` to the number of modes an rfft of that length
    actually has.
    """

    def __init__(
        self,
        n_points,
        in_channels=1,
        out_channels=1,
        modes=16,
        width=64,
    ):
        super(FNOOperator, self).__init__()
        self.n_points = n_points

        # rfft of a length-N signal has N//2 + 1 coefficients; keeping more
        # modes than that would index past the end of the spectrum.
        max_modes = n_points // 2 + 1
        self.modes = min(modes, max_modes)

        self.fno = FNO1d(
            modes=self.modes,
            width=width,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, u0):
        return self.fno(u0)
