"""
Fourier Neural Operator (FNO), 2D - used as the geometry-aware baseline on
the torus.

Reference implementation, kept faithful to the official code released with

    Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar,
    "Fourier Neural Operator for Parametric Partial Differential
    Equations", ICLR 2021 (arXiv:2010.08895)

    https://github.com/neuraloperator/neuraloperator  (fourier_2d.py)

SpectralConv2d and FNO2d reproduce the reference architecture: each Fourier
layer keeps the lowest `modes1` x `modes2` coefficients of rfft2 (both the
positive and negative frequencies along the first axis, hence two weight
tensors), multiplies them by learned complex weights and transforms back, in
parallel with a pointwise (1x1) convolution. Four layers with GELU, a lifting
Linear and a two-layer pointwise decoder; the normalised (x, y) grid is
concatenated before lifting. Reference defaults: modes=12, width=32.

One deliberate deviation: the reference pads the grid (padding=9) because
its benchmark domains are not periodic. The torus *is* periodic in both
angles, which is exactly what the FFT assumes, so padding is disabled.

TorusFNO2dOperator at the bottom is the only addition: it maps the torus
mesh vertices (trimesh.creation.torus lays them out on an exact
major x minor angle grid) onto that periodic 2D grid and back, so FNO2d sees
the torus's true geometry rather than a 1D vertex sequence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: FFT -> keep the lowest modes -> multiply by learned
    complex weights -> inverse FFT.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2,
                dtype=torch.cfloat,
            )
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator (reference architecture, no padding).

    Input [batch, x, y, in_channels]; the normalised (x, y) grid is
    concatenated, so the lifting layer sees in_channels + 2.
    Output [batch, x, y, out_channels].
    """

    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(in_channels + 2, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

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

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


# ---------------------------------------------------------------------------
# Benchmarking wrapper
#
# Not part of the reference FNO: maps torus mesh vertices onto the periodic
# (major angle x minor angle) grid they were generated on, runs FNO2d there,
# and maps back. No interpolation is involved - it is a permutation.
# ---------------------------------------------------------------------------


def torus_grid_index(vertices, n_major, n_minor, major_radius):
    """
    Grid position (i * n_minor + j) of every vertex, where i indexes the
    major angle and j the minor (tube) angle.

    Raises if the vertices do not form a complete n_major x n_minor grid,
    i.e. if the mesh was not built by trimesh.creation.torus with these
    section counts.
    """
    vertices = np.asarray(vertices, dtype=float)

    major = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2 * np.pi)
    rho = np.hypot(vertices[:, 0], vertices[:, 1])
    minor = np.mod(np.arctan2(vertices[:, 2], rho - major_radius), 2 * np.pi)

    i = np.round(major / (2 * np.pi) * n_major).astype(int) % n_major
    j = np.round(minor / (2 * np.pi) * n_minor).astype(int) % n_minor
    index = i * n_minor + j

    if len(vertices) != n_major * n_minor or len(np.unique(index)) != len(index):
        raise ValueError(
            "torus vertices do not form a complete "
            f"{n_major} x {n_minor} angle grid"
        )
    return index


class TorusFNO2dOperator(nn.Module):
    """
    FNO2d on the torus's own periodic angle grid.

    Expects [batch, n_vertices, in_channels] in mesh vertex order and
    returns [batch, n_vertices, out_channels] in the same order.
    """

    def __init__(
        self,
        vertices,
        n_major,
        n_minor,
        major_radius,
        in_channels=1,
        out_channels=1,
        modes=12,
        width=32,
    ):
        super(TorusFNO2dOperator, self).__init__()
        self.n_major = n_major
        self.n_minor = n_minor

        index = torus_grid_index(vertices, n_major, n_minor, major_radius)

        # vertex_at_grid[g] = vertex sitting at grid position g
        vertex_at_grid = np.empty_like(index)
        vertex_at_grid[index] = np.arange(len(index))

        self.register_buffer("grid_index", torch.as_tensor(index, dtype=torch.long))
        self.register_buffer(
            "vertex_at_grid", torch.as_tensor(vertex_at_grid, dtype=torch.long)
        )

        # rfft2 keeps n // 2 + 1 coefficients on the last axis; the first
        # axis stores +/- frequencies, so it can hold at most n // 2 of each.
        modes1 = min(modes, n_major // 2)
        modes2 = min(modes, n_minor // 2 + 1)

        self.fno = FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, u0):
        batch, _, channels = u0.shape

        grid = u0[:, self.vertex_at_grid].reshape(
            batch, self.n_major, self.n_minor, channels
        )
        out = self.fno(grid)
        out = out.reshape(batch, self.n_major * self.n_minor, -1)

        return out[:, self.grid_index]
