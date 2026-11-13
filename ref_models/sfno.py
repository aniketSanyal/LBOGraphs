"""
Spherical Fourier Neural Operator (SFNO-style), used as the geometry-aware
baseline on the sphere.

Follows the idea of

    Bonev, Kurth, Hundt, Pathak, Baust, Kashinath, Anandkumar,
    "Spherical Fourier Neural Operators: Learning Stable Dynamics on the
    Sphere", ICML 2023 (arXiv:2306.03838)

    https://github.com/NVIDIA/torch-harmonics

SFNO replaces the FFT of FNO with the spherical harmonic transform (SHT).
This file keeps the FNO1d/FNO2d layer structure used by the other baselines
in this benchmark (lift -> 4 x [spectral conv + pointwise 1x1, GELU] ->
two-layer pointwise decoder, grid coordinates concatenated before lifting)
and swaps the spectral convolution for an SHT-domain one, so the comparison
isolates the basis. It is a deliberate simplification of the official
SFNO network, which additionally uses layer norms and per-block MLPs.

Spectral weights use the Driscoll-Healy form: one channel-mixing matrix per
degree l, shared by all orders m. This is the spherical convolution theorem
(convolution with a zonal kernel), which makes the layer rotation-
equivariant, as in the reference "driscoll-healy" operator type. Degrees
above `modes` are dropped, mirroring FNO's mode truncation.

The SHT is computed by quadrature on a Gauss-Legendre latitude x equispaced
longitude grid (exact for band-limited fields), with real spherical
harmonics built from the standard stable recurrence for fully normalised
associated Legendre functions - no torch-harmonics or scipy.special
dependency. At this resolution a dense matrix product is cheaper than the
FFT + Legendre factorisation torch-harmonics uses.

SphereSFNOOperator at the bottom adapts it to data living on sphere mesh
vertices: inputs are mapped vertex -> grid by a least-squares spherical
harmonic fit at the vertices, and outputs grid -> vertex by evaluating their
spherical harmonic expansion at the vertices.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spherical harmonics
# ---------------------------------------------------------------------------


def gauss_legendre_grid(nlat):
    """
    Gauss-Legendre latitudes x equispaced longitudes, nlon = 2 * nlat.

    Returns colatitude theta [nlat], longitude phi [nlon] and the quadrature
    weights [nlat * nlon] (summing to 4 pi), flattened latitude-major.
    """
    cos_theta, lat_weights = np.polynomial.legendre.leggauss(nlat)
    theta = np.arccos(cos_theta)
    nlon = 2 * nlat
    phi = 2 * np.pi * np.arange(nlon) / nlon

    weights = np.outer(lat_weights, np.full(nlon, 2 * np.pi / nlon)).ravel()
    return theta, phi, weights


def real_spherical_harmonics(lmax, theta, phi):
    """
    Real, orthonormal spherical harmonics up to degree lmax at the points
    (theta, phi) (same length, colatitude/longitude).

    Returns Y [n_points, (lmax + 1)^2] ordered by (l, m), m = -l..l, and the
    degree l of every column.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    x, s = np.cos(theta), np.sin(theta)

    # Fully normalised associated Legendre functions Pbar[l][m](x), such that
    # Pbar_l^m(cos theta) * {1, sqrt2 cos m phi, sqrt2 sin m phi} is
    # orthonormal on the unit sphere.
    P = {(0, 0): np.full_like(x, np.sqrt(1.0 / (4.0 * np.pi)))}
    for m in range(1, lmax + 1):
        P[(m, m)] = -np.sqrt((2 * m + 1) / (2 * m)) * s * P[(m - 1, m - 1)]
    for m in range(0, lmax):
        P[(m + 1, m)] = np.sqrt(2 * m + 3) * x * P[(m, m)]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = np.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = np.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[(l, m)] = a * (x * P[(l - 1, m)] - b * P[(l - 2, m)])

    columns, degrees = [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if m < 0:
                columns.append(np.sqrt(2) * P[(l, -m)] * np.sin(-m * phi))
            elif m == 0:
                columns.append(P[(l, 0)])
            else:
                columns.append(np.sqrt(2) * P[(l, m)] * np.cos(m * phi))
            degrees.append(l)

    return np.stack(columns, axis=1), np.asarray(degrees)


# ---------------------------------------------------------------------------
# SFNO
# ---------------------------------------------------------------------------


class SpectralConvS2(nn.Module):
    """
    SHT -> per-degree (Driscoll-Healy) channel mixing for l <= modes ->
    inverse SHT. Operates channels-last on [batch, n_grid, channels].
    """

    def __init__(self, in_channels, out_channels, modes, Y, quad_weights, degrees):
        super(SpectralConvS2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        keep = degrees <= modes
        Y = torch.as_tensor(Y[:, keep], dtype=torch.float32)
        w = torch.as_tensor(quad_weights, dtype=torch.float32)

        self.register_buffer("analysis", (Y * w[:, None]).T.contiguous())   # [K, G]
        self.register_buffer("synthesis", Y.contiguous())                   # [G, K]
        self.register_buffer(
            "degree", torch.as_tensor(degrees[keep], dtype=torch.long)
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes + 1)
        )

    def forward(self, x):
        coefficients = torch.einsum("kg,bgc->bkc", self.analysis, x)
        weights = self.weights[:, :, self.degree]                 # [C_in, C_out, K]
        coefficients = torch.einsum("bkc,cok->bko", coefficients, weights)
        return torch.einsum("gk,bko->bgo", self.synthesis, coefficients)


class SFNO(nn.Module):
    """
    FNO layer structure with the FFT replaced by the SHT.

    Input [batch, n_grid, in_channels] on the Gauss-Legendre grid; the grid
    point xyz coordinates are concatenated before lifting (the spherical
    analogue of FNO's get_grid). Output [batch, n_grid, out_channels].
    """

    def __init__(self, nlat, modes, width, in_channels=1, out_channels=1):
        super(SFNO, self).__init__()

        self.width = width
        theta, phi, quad_weights = gauss_legendre_grid(nlat)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        theta_grid, phi_grid = theta_grid.ravel(), phi_grid.ravel()

        # lmax = nlat - 1 is the highest degree the quadrature resolves
        Y, degrees = real_spherical_harmonics(nlat - 1, theta_grid, phi_grid)
        modes = min(modes, nlat - 1)

        xyz = np.stack(
            [np.sin(theta_grid) * np.cos(phi_grid),
             np.sin(theta_grid) * np.sin(phi_grid),
             np.cos(theta_grid)],
            axis=-1,
        )
        self.register_buffer("grid_xyz", torch.as_tensor(xyz, dtype=torch.float32))

        self.fc0 = nn.Linear(in_channels + 3, self.width)

        self.conv0 = SpectralConvS2(self.width, self.width, modes, Y, quad_weights, degrees)
        self.conv1 = SpectralConvS2(self.width, self.width, modes, Y, quad_weights, degrees)
        self.conv2 = SpectralConvS2(self.width, self.width, modes, Y, quad_weights, degrees)
        self.conv3 = SpectralConvS2(self.width, self.width, modes, Y, quad_weights, degrees)
        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.grid_xyz.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Benchmarking wrapper: sphere mesh vertices <-> Gauss-Legendre grid
# ---------------------------------------------------------------------------


def default_fit_degree(n_vertices, oversampling=1.5):
    """Largest degree L whose (L + 1)^2 harmonics are fitted from at least
    `oversampling` vertices each, keeping the least-squares fit well posed."""
    return int(np.floor(np.sqrt(n_vertices / oversampling))) - 1


class SphereSFNOOperator(nn.Module):
    """
    SFNO for fields sampled on sphere mesh vertices.

    Expects [batch, n_vertices, in_channels] and returns
    [batch, n_vertices, out_channels], in mesh vertex order.

    vertex -> grid : least-squares fit of spherical harmonics up to degree
                     fit_degree at the vertices, evaluated on the grid
                     (exact for fields band-limited to fit_degree; plain
                     barycentric interpolation on the faceted mesh loses
                     ~1.6% even on degree-3 fields at this resolution)
    grid -> vertex : spherical harmonic analysis on the grid, synthesis at
                     the vertices (degrees up to nlat - 1)
    """

    def __init__(
        self,
        vertices,
        faces,
        in_channels=1,
        out_channels=1,
        nlat=32,
        modes=16,
        width=64,
        fit_degree=None,
    ):
        super(SphereSFNOOperator, self).__init__()

        vertices = np.asarray(vertices, dtype=np.float64)
        directions = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        vertex_theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
        vertex_phi = np.arctan2(directions[:, 1], directions[:, 0])

        theta, phi, quad_weights = gauss_legendre_grid(nlat)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        theta_grid, phi_grid = theta_grid.ravel(), phi_grid.ravel()
        lmax = nlat - 1
        Y_grid, degrees = real_spherical_harmonics(lmax, theta_grid, phi_grid)
        Y_vertex, _ = real_spherical_harmonics(lmax, vertex_theta, vertex_phi)

        if fit_degree is None:
            fit_degree = default_fit_degree(len(vertices))
        self.fit_degree = min(fit_degree, lmax)
        fit = degrees <= self.fit_degree

        to_grid = Y_grid[:, fit] @ np.linalg.pinv(Y_vertex[:, fit])
        to_vertex = Y_vertex @ (Y_grid * quad_weights[:, None]).T

        self.register_buffer("to_grid", torch.as_tensor(to_grid, dtype=torch.float32))
        self.register_buffer("to_vertex", torch.as_tensor(to_vertex, dtype=torch.float32))

        self.sfno = SFNO(
            nlat=nlat,
            modes=modes,
            width=width,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def round_trip(self, u):
        """vertex -> grid -> vertex without the network: the interpolation
        error floor this baseline cannot go below."""
        grid = torch.einsum("gv,bvc->bgc", self.to_grid, u)
        return torch.einsum("vg,bgc->bvc", self.to_vertex, grid)

    def forward(self, u0):
        grid = torch.einsum("gv,bvc->bgc", self.to_grid, u0)
        out = self.sfno(grid)
        return torch.einsum("vg,bgc->bvc", self.to_vertex, out)
