"""
Laplace Neural Operator (LNO), 1D.

Reference implementation, kept faithful to the official code released with

    Cao, Goswami, Karniadakis, "Laplace neural operator for solving
    differential equations", Nature Machine Intelligence 6, 631-640 (2024)
    (arXiv:2303.10528)

    https://github.com/qianyingcao/Laplace-Neural-Operator
    (1D_Pendulum_c05/main.py, 1D_Duffing_c05/main.py, ...)

The PR (pole-residue) layer below reproduces the reference architecture.
Unlike FNO, which multiplies truncated Fourier coefficients by a learned
weight, LNO parameterizes the kernel in the Laplace domain in pole-residue
form K(s) = sum_n beta_n / (s - mu_n), with learnable complex poles
`weights_pole` (mu) and residues `weights_residue` (beta). Multiplying by
the transformed input and applying residue calculus splits the response
into two parts, exactly as in the paper:

    x1 : the steady-state response, recovered by inverse FFT of the
         residues taken at the input poles i*omega;
    x2 : the transient response, reconstructed directly as a sum of the
         learned exponentials exp(mu_n * t).

Summing them gives a response that can represent decaying/growing
transients, which a pure Fourier basis cannot.

Two deviations from the reference file are unavoidable and deliberate:

  * The reference reads the time axis from a module-level global
    (`grid_x_train`, loaded from the benchmark's .mat file) and hardcodes
    `.cuda()`. Here the time axis is passed to the constructor and stored
    as a buffer, so the layer is self-contained and device-agnostic. It is
    still the physical time coordinate the exponentials exp(mu*t) are
    evaluated on, which is what the reference used it for.
  * `torch.rand` pole initialization is kept, but see LNOOperator for the
    note on why the time axis must match the data.

LNOOperator at the bottom is the only addition: a thin wrapper that adapts
LNO1d to the initial-condition -> final-state mapping produced by
equations.py, so it can be benchmarked against other architectures on the
same data. It mirrors CNOOperator / FNOOperator.
"""

import numpy as np
import torch
import torch.nn as nn


class PR(nn.Module):
    """
    Pole-residue layer.

    Learns complex poles (mu) and residues (beta) defining the Laplace
    domain kernel K(s) = sum_n beta_n / (s - mu_n), and returns the sum of
    the steady-state and transient responses.

    t : [n_points] tensor, the physical time/space coordinate the
        transient exponentials are evaluated on.
    """

    def __init__(self, in_channels, out_channels, modes1, t):
        super(PR, self).__init__()

        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)

        self.weights_pole = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, dtype=torch.cfloat
            )
        )
        self.weights_residue = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, dtype=torch.cfloat
            )
        )

        # Time axis: a buffer rather than the reference's module-level
        # global, so the layer moves with .to(device) like any other module.
        self.register_buffer("t", torch.as_tensor(t, dtype=torch.float))

    def output_PR(self, lambda1, alpha, weights_pole, weights_residue):
        # Hw[k, i, o, n] = beta_{i,o,n} / (lambda_k - mu_{i,o,n})
        term1 = torch.div(1, torch.sub(lambda1, weights_pole))
        Hw = weights_residue * term1
        Pk = -Hw

        # Residues at the input poles i*omega -> steady-state part
        output_residue1 = torch.einsum("bix,xiok->box", alpha, Hw)
        # Residues at the kernel poles mu_n -> transient part
        output_residue2 = torch.einsum("bix,xiok->bok", alpha, Pk)

        return output_residue1, output_residue2

    def forward(self, x):
        t = self.t
        dt = (t[1] - t[0]).item()

        alpha = torch.fft.fft(x)

        lambda0 = torch.fft.fftfreq(t.shape[0], dt).to(x.device) * 2 * np.pi * 1j
        lambda1 = lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        output_residue1, output_residue2 = self.output_PR(
            lambda1, alpha, self.weights_pole, self.weights_residue
        )

        # Steady-state response: inverse transform of the input-pole residues
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)

        # Transient response: sum of learned exponentials exp(mu_n * t)
        term1 = torch.einsum(
            "bix,kz->bixz",
            self.weights_pole,
            t.type(torch.complex64).reshape(1, -1),
        )
        term2 = torch.exp(term1)
        x2 = torch.einsum("bix,ioxz->boz", output_residue2, term2)
        x2 = torch.real(x2)
        x2 = x2 / x.size(-1)

        return x1 + x2


class LNO1d(nn.Module):
    """
    1D Laplace Neural Operator (reference architecture).

    modes : number of poles/residues in the learned Laplace kernel
    width : lifted channel width
    t     : [n_points] physical coordinate vector

    A single pole-residue layer summed with a parallel pointwise
    convolution, then a two-layer pointwise decoder with a sine
    activation - all as in the reference. Reference defaults for the 1D
    ODE benchmarks are width=4 with modes=8..16, i.e. a deliberately tiny
    model; the expressiveness comes from the pole-residue basis rather
    than from depth.

    Following the Pendulum benchmark, the normalized coordinate grid is
    concatenated as an extra input channel before lifting.
    """

    def __init__(self, width, modes, t, in_channels=1, out_channels=1):
        super(LNO1d, self).__init__()

        self.width = width
        self.modes1 = modes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc0 = nn.Linear(in_channels + 1, self.width)

        self.conv0 = PR(self.width, self.width, self.modes1, t)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

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

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# ---------------------------------------------------------------------------
# Benchmarking wrapper
#
# Not part of the reference LNO: this adapts LNO1d to the state-evolution
# task defined by equations.py, mapping an initial field u(0) on a domain to
# the final field u(T), so LNO can be compared against other architectures on
# identical data. Mirrors ref_models/cno.py and ref_models/fno.py.
# ---------------------------------------------------------------------------


class LNOOperator(nn.Module):
    """
    Maps an initial state to a final state with LNO1d.

    Expects inputs shaped [batch, n_points, in_channels] (the layout
    produced by equations.py solutions) and returns
    [batch, n_points, out_channels] - LNO1d already uses this
    channels-last convention, so no transposition is needed.

    coords : optional [n_points] array giving the physical coordinate of
             each point (e.g. domain.points[:, 0] for a 1D GridDomain, or
             the time vector when learning an ODE trajectory). The
             pole-residue layer evaluates exp(mu * coords) on it, so its
             scale sets the decay rates the model can represent. Defaults
             to a uniform grid on [0, 1] to match the reference's
             normalized axis.
    """

    def __init__(
        self,
        n_points,
        in_channels=1,
        out_channels=1,
        modes=16,
        width=4,
        coords=None,
    ):
        super(LNOOperator, self).__init__()
        self.n_points = n_points

        if coords is None:
            coords = np.linspace(0.0, 1.0, n_points)
        coords = np.asarray(coords, dtype=float).reshape(-1)

        if coords.shape[0] != n_points:
            raise ValueError(
                f"coords has {coords.shape[0]} entries but n_points is "
                f"{n_points}"
            )

        self.modes = modes

        self.lno = LNO1d(
            width=width,
            modes=modes,
            t=coords,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, u0):
        return self.lno(u0)
