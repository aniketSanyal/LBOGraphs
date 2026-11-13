import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp


class Equation:
    """
    Base class for a first-order ODE system written as

        dstate/dt = f(t, state)

    Subclasses implement `evolution` (the right-hand side) and
    `set_initial_condition` (how to build/store a starting state),
    so different equations can be swapped in and out while testing
    architectures against a common interface.
    """

    def __init__(self):
        self.state0 = None

    def evolution(self, t, state):
        raise NotImplementedError

    def set_initial_condition(self, state0):
        raise NotImplementedError

    def get_initial_condition(self):
        if self.state0 is None:
            raise ValueError(
                "Initial condition not set. "
                "Call set_initial_condition first."
            )
        return self.state0

    def solve(
        self,
        t_span,
        t_eval=None,
        state0=None,
        method="RK45",
        **kwargs,
    ):
        """
        Integrate this equation's `evolution` with
        scipy.integrate.solve_ivp.

        state0 defaults to the stored initial condition
        (set via set_initial_condition) if not given.
        """
        if state0 is None:
            state0 = self.get_initial_condition()

        return solve_ivp(
            self.evolution,
            t_span,
            state0,
            t_eval=t_eval,
            method=method,
            **kwargs,
        )


class DrivenPendulum(Equation):
    """
    Driven, damped pendulum:

        xddot + c * xdot + sin(x) = f(t)

    written as the first-order system state = (x, xdot):

        xdot  = xdot
        xddot = f(t) - c * xdot - sin(x)

    forcing is any callable f(t); pass None for the unforced pendulum.
    """

    def __init__(self, c, forcing=None):
        super().__init__()
        self.c = c
        self.forcing = forcing

    def f(self, t):
        if self.forcing is None:
            return 0.0
        return self.forcing(t)

    def evolution(self, t, state):
        x, xdot = state

        xddot = self.f(t) - self.c * xdot - np.sin(x)

        return np.array([xdot, xddot])

    def set_initial_condition(self, x0, xdot0):
        self.state0 = np.array([x0, xdot0])
        return self.state0


class DuffingOscillator(Equation):
    """
    Duffing oscillator:

        xddot + c * xdot + x + x**3 = f(t)

    written as the first-order system state = (x, xdot):

        xdot  = xdot
        xddot = f(t) - c * xdot - x - x**3

    forcing is any callable f(t); pass None for the unforced case.
    """

    def __init__(self, c, forcing=None):
        super().__init__()
        self.c = c
        self.forcing = forcing

    def f(self, t):
        if self.forcing is None:
            return 0.0
        return self.forcing(t)

    def evolution(self, t, state):
        x, xdot = state

        xddot = self.f(t) - self.c * xdot - x - x**3

        return np.array([xdot, xddot])

    def set_initial_condition(self, x0, xdot0):
        self.state0 = np.array([x0, xdot0])
        return self.state0


class LorenzSystem(Equation):
    """
    Driven Lorenz system, state = (x, y, z):

        xdot = 10 * (y - x)
        ydot = x * (rho - z) - y
        zdot = x * y - (8 / 3) * z - f(t)

    forcing is any callable f(t); pass None for the unforced system.
    """

    def __init__(self, rho, forcing=None):
        super().__init__()
        self.rho = rho
        self.forcing = forcing

    def f(self, t):
        if self.forcing is None:
            return 0.0
        return self.forcing(t)

    def evolution(self, t, state):
        x, y, z = state

        xdot = 10.0 * (y - x)
        ydot = x * (self.rho - z) - y
        zdot = x * y - (8.0 / 3.0) * z - self.f(t)

        return np.array([xdot, ydot, zdot])

    def set_initial_condition(self, x0, y0, z0):
        self.state0 = np.array([x0, y0, z0])
        return self.state0


# ---------------------------------------------------------------------------
# Spatial domains
#
# A Domain discretizes a manifold/region into a fixed set of points and
# exposes the discrete Laplacian acting on functions sampled at those
# points. Spatial PDEs below (diffusion, reaction-diffusion, beam) are
# written purely in terms of `domain.laplacian`, so the same equation runs
# unchanged on a line, a regular grid, a sphere, a torus, or an arbitrary
# mesh/graph - only the Domain changes.
# ---------------------------------------------------------------------------


class Domain:
    """
    Base class for a discretized spatial domain.

    Subclasses populate:
        self.points     [n_points, dim] coordinates of the discretization
        self.n_points    number of points
        self.laplacian   [n_points, n_points] discrete Laplacian (sparse),
                          such that (laplacian @ y) approximates the
                          continuous Laplacian of y at each point
    """

    points = None
    n_points = None
    laplacian = None

    def zeros(self):
        return np.zeros(self.n_points)


class GridDomain(Domain):
    """
    Regular N-dimensional grid on [0, L_1] x ... x [0, L_d], discretized
    with `shape` points per axis and a standard second-order central
    finite-difference stencil for the Laplacian.

    Homogeneous Dirichlet boundary conditions (y = 0 on the boundary,
    boundary = {x : at least one coordinate is an endpoint}) are enforced
    by eliminating boundary degrees of freedom entirely: `self.points`,
    `self.n_points`, and `self.laplacian` all refer to *interior* grid
    points only. This is what makes u = 0 on the boundary exact - there
    is no boundary state left to be nonzero, so a bad initial condition
    or nonzero forcing at the boundary simply cannot leak in. Naively
    zeroing boundary rows of each 1D operator before summing them into
    the N-D Laplacian does *not* work: at a non-corner boundary point
    only one axis contributes a zero row, so the other axis derivative
    still evolves that point away from zero.

    The full (uneliminated) grid is kept as `self.full_points` /
    `self.interior_mask` for embedding an interior solution back onto
    the full grid, e.g. for plotting.
    """

    def __init__(self, lengths, shape):
        lengths = np.atleast_1d(np.asarray(lengths, dtype=float))
        shape = np.atleast_1d(np.asarray(shape, dtype=int))

        if lengths.shape != shape.shape:
            raise ValueError("lengths and shape must have the same length")

        self.dim = len(shape)
        self.shape = tuple(shape)
        self.lengths = lengths
        self.spacing = lengths / (shape - 1)

        axes = [
            np.linspace(0.0, length, n)
            for length, n in zip(lengths, shape)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.full_points = np.stack([m.ravel() for m in mesh], axis=-1)

        # Boundary = any coordinate equal to 0 or its axis length.
        on_boundary = np.zeros(self.full_points.shape[0], dtype=bool)
        for axis in range(self.dim):
            coord = self.full_points[:, axis]
            on_boundary |= np.isclose(coord, 0.0)
            on_boundary |= np.isclose(coord, lengths[axis])

        self.interior_mask = ~on_boundary
        self.interior_index = np.flatnonzero(self.interior_mask)

        full_laplacian = self._build_full_laplacian()

        # Eliminate boundary DOFs: keep only interior rows/columns.
        self.points = self.full_points[self.interior_mask]
        self.n_points = self.points.shape[0]
        self.laplacian = full_laplacian[
            self.interior_index[:, None], self.interior_index
        ].tocsr()

    def embed(self, interior_values):
        """
        Place an interior-only field back onto the full grid, filling
        the boundary with zeros (matching the homogeneous Dirichlet
        condition). Returns an array of shape
        (*self.shape, *interior_values.shape[1:]).
        """
        interior_values = np.asarray(interior_values)
        full = np.zeros(
            (self.full_points.shape[0],) + interior_values.shape[1:]
        )
        full[self.interior_mask] = interior_values
        return full.reshape(self.shape + interior_values.shape[1:])

    def _axis_second_derivative(self, n, h):
        main = -2.0 * np.ones(n)
        off = np.ones(n - 1)
        L = sp.diags(
            [off, main, off],
            offsets=[-1, 0, 1],
            format="csr",
        ) / h**2
        return L

    def _build_full_laplacian(self):
        # Sum of 1D second-derivative operators lifted to the full tensor
        # grid via Kronecker products, i.e. the standard finite-difference
        # discretization of the N-D Laplacian on a separable grid. Boundary
        # elimination happens afterwards, on the full operator, so it sees
        # every point's true set of active neighbors.
        d = self.dim
        n_full = self.full_points.shape[0]
        L = sp.csr_matrix((n_full, n_full))

        for axis in range(d):
            axis_op = self._axis_second_derivative(
                self.shape[axis], self.spacing[axis]
            )

            factors = [
                axis_op if i == axis else sp.identity(
                    self.shape[i], format="csr"
                )
                for i in range(d)
            ]

            term = factors[0]
            for factor in factors[1:]:
                term = sp.kron(term, factor, format="csr")

            L = L + term

        return L.tocsr()


class MeshDomain(Domain):
    """
    Discretization on an arbitrary triangle mesh (sphere, torus, any
    surface) using the cotangent Laplace-Beltrami operator via
    `robust_laplacian.mesh_laplacian`, i.e. L = M^-1 S where S is the
    stiffness matrix and M the (lumped, diagonal) mass matrix. This is
    the geometrically accurate discretization used elsewhere in this
    project for meshes, as opposed to a naive graph/grid Laplacian.

    vertices: [n, 3] array of vertex positions
    faces:    [m, 3] array of triangle vertex indices
    """

    def __init__(self, vertices, faces):
        import robust_laplacian

        self.points = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces)
        self.n_points = self.points.shape[0]

        S, M = robust_laplacian.mesh_laplacian(self.points, self.faces)

        self.stiffness = S.tocsr()
        self.mass = M.tocsr()
        self.mass_diagonal = np.asarray(M.diagonal())

        # robust_laplacian returns S as the positive semi-definite
        # stiffness matrix (i.e. S = -M @ (Laplace-Beltrami)), so
        # M^-1 S has eigenvalues +lambda_k >= 0. Negate to match this
        # module's convention that `laplacian @ y` approximates the
        # analyst's (negative semi-definite) Laplacian, consistent
        # with GridDomain and GraphDomain.
        M_inv = sp.diags(1.0 / self.mass_diagonal)
        self.laplacian = (-(M_inv @ self.stiffness)).tocsr()

    @classmethod
    def from_trimesh(cls, mesh):
        return cls(mesh.vertices, mesh.faces)

    @classmethod
    def sphere(cls, subdivisions=3, radius=1.0):
        import trimesh

        mesh = trimesh.creation.icosphere(
            subdivisions=subdivisions, radius=radius
        )
        return cls.from_trimesh(mesh)

    @classmethod
    def torus(cls, major_radius=1.0, minor_radius=0.3, **kwargs):
        import trimesh

        mesh = trimesh.creation.torus(
            major_radius=major_radius,
            minor_radius=minor_radius,
            **kwargs,
        )
        return cls.from_trimesh(mesh)


class GraphDomain(Domain):
    """
    Discretization on an arbitrary graph / point cloud with precomputed
    edges, using the (optionally weighted) graph Laplacian L = D - A.

    points: [n, d] coordinates (used for plotting/forcing functions;
            not required for the Laplacian itself)
    edges:  [m, 2] array of vertex index pairs
    weights: optional [m] edge weights (defaults to all ones)
    """

    def __init__(self, points, edges, weights=None):
        self.points = np.asarray(points, dtype=float)
        self.n_points = self.points.shape[0]

        edges = np.asarray(edges)
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])

        if weights is None:
            data = np.ones(len(src))
        else:
            data = np.concatenate([weights, weights])

        A = sp.coo_matrix(
            (data, (src, dst)),
            shape=(self.n_points, self.n_points),
        ).tocsr()

        D = sp.diags(np.asarray(A.sum(axis=1)).ravel())

        # Graph Laplacian D - A approximates -Laplacian (positive
        # semi-definite), so negate to match the sign convention of
        # laplacian @ y ~= continuous Laplacian of y used elsewhere.
        self.laplacian = -(D - A).tocsr()


# ---------------------------------------------------------------------------
# Spatial PDEs (domain-agnostic: same equation, any Domain)
# ---------------------------------------------------------------------------


class SpatialEquation(Equation):
    """
    Base class for PDEs of the form dy/dt = g(t, y, domain.laplacian @ y),
    solved by the method of lines: the spatial operator comes from a
    Domain (grid, mesh, graph, ...), turning the PDE into an ODE system
    with one state entry per domain point, which plugs straight into
    Equation.solve / scipy.integrate.solve_ivp.

    forcing is any callable f(points, t); pass None for zero forcing.
    """

    def __init__(self, domain, forcing=None):
        super().__init__()
        self.domain = domain
        self.forcing = forcing

    def f(self, t):
        if self.forcing is None:
            return 0.0
        return self.forcing(self.domain.points, t)

    def set_initial_condition(self, y0):
        self.state0 = np.asarray(y0)
        return self.state0


class DiffusionEquation(SpatialEquation):
    """
    Diffusion equation on an arbitrary domain:

        dy/dt = D * laplacian(y) + f(points, t)
    """

    def __init__(self, domain, D, forcing=None):
        super().__init__(domain, forcing)
        self.D = D

    def evolution(self, t, state):
        y = state
        return self.D * (self.domain.laplacian @ y) + self.f(t)


class ReactionDiffusion(SpatialEquation):
    """
    Reaction-diffusion equation on an arbitrary domain:

        dy/dt = D * laplacian(y) + k * y**2 + f(points, t)
    """

    def __init__(self, domain, D, k, forcing=None):
        super().__init__(domain, forcing)
        self.D = D
        self.k = k

    def evolution(self, t, state):
        y = state
        return (
            self.D * (self.domain.laplacian @ y)
            + self.k * y**2
            + self.f(t)
        )


class BeamEquation(SpatialEquation):
    """
    Euler-Bernoulli-type beam on an arbitrary domain, state = (w, wdot)
    stacked as one vector:

        rho_A * wddot = f(points, t) - EI * biharmonic(w)

    where biharmonic = laplacian @ laplacian, written as the first-order
    system:

        wdot  = wdot
        wddot = (f(points, t) - EI * biharmonic @ w) / rho_A
    """

    def __init__(self, domain, EI, rho_A, forcing=None):
        super().__init__(domain, forcing)
        self.EI = EI
        self.rho_A = rho_A
        self.biharmonic = domain.laplacian @ domain.laplacian

    def evolution(self, t, state):
        n = self.domain.n_points
        w, wdot = state[:n], state[n:]

        wddot = (
            self.f(t) - self.EI * (self.biharmonic @ w)
        ) / self.rho_A

        return np.concatenate([wdot, wddot])

    def set_initial_condition(self, w0, wdot0):
        self.state0 = np.concatenate(
            [np.asarray(w0), np.asarray(wdot0)]
        )
        return self.state0
