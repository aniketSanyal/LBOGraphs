import numpy as np
from typing import Sequence
import torch
import torch.nn as nn

ArrayLike = torch.Tensor | np.ndarray


class HeatKernel(nn.Module):
    def __init__(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        mass: ArrayLike,
        in_channels: int,
        out_channels: int,
        diffusion_times: Sequence[float] | ArrayLike 
    ):
        """
        self notes
        
        eigenvalues: these are the eigenvalues of the Laplacian
        
        eigenvectors: eigenvectors of the Laplacian
        
        mass: due to uneven discretization/ uneven mesh, different vertices have different mass so reweighting them - ask if needed
        
        n_basis: the basis functions we are using
        """
        super().__init__()
        self.n_modes = eigenvalues.numel()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.register_buffer(
            "eigenvalues",
            torch.as_tensor(eigenvalues.flatten().clamp_min(0.0), dtype= torch.float32)
        )
        self.register_buffer(
            "Phi",
            torch.as_tensor(eigenvectors, dtype=torch.float32),
        )
        self.register_buffer(
            "mass",
            torch.as_tensor(mass, dtype=torch.float32),
        )
        
        times = torch.as_tensor(
            diffusion_times,
            dtype=torch.float32,
        ).flatten()
        self.register_buffer(
            "times",
            times
        )
        self.n_diffusion_scales = times.numel()

        self.coefficients = nn.Parameter(
            torch.empty(
                self.n_diffusion_scales,
                in_channels,
                out_channels
            )
        )
        
        for time_scale in range(self.n_diffusion_scales):
            nn.init.xavier_uniform_(self.coefficients[time_scale])    
            
    def exp_basis(self) -> torch.Tensor:
        """
        E[r, k] = exp(-t_r * \\lambda_k)
        """
        
        return torch.exp(
            -self.times[:, None] * self.eigenvalues[None, :]
        )
        
    def heat_kernel_matrices(self) -> torch.Tensor:
        """
        Returns:
            K: [R, V, V]

            K[r,v,u] = sum_k
                Phi[v,k] exp(-t_r lambda_k) Phi[u,k]
        """
        # [R, m]
        E = self.exp_basis()  

        return torch.einsum(
            "vk,rk,uk->rvu",
            self.Phi,
            E,
            self.Phi,
        )
    
    
    def forward(self, f: torch.Tensor) -> torch.Tensor :
        
        #first, projection of f onto basis func
        #f_hat = phi^T M f
        #M - just mass func for reweighting vertices
        # [B,V,C_in] -> [B,m,C_in]
        f_hat = torch.einsum(
            "vm, bvc, v->bmc",
            self.Phi,
            f,
            self.mass
        )
        
        E = self.exp_basis()
        
        #multiply exponential basis with f_hat and the weight matrix
        # y_hat[b,k,o]
        #   = sum_{r,c} E[r,k] f_hat[b,k,c] W[r,c,o]
        # [B,m,C_out]
        y_hat = torch.einsum(
            "rm, bmc, rco-> bmo",
            E,
            f_hat,
            self.coefficients
        )
        
        #project back using basis
        # [B,m,C_out] -> [B,V,C_out]
        y = torch.einsum(
            "vm, bmo -> bvo",
            self.Phi,
            y_hat
        )
        
        return y
        
        
class IntegralOperator(nn.Module):
    def __init__(
        self,
        kernel: HeatKernel,
        hidden_channels: int
    ) -> None:
        
        super().__init__()
        self.kernel = kernel
        in_channels = kernel.in_channels
        
        self.phi = nn.Sequential(
            nn.Linear(
                2*in_channels,
                hidden_channels
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                in_channels
            )
        )  
        
    def forward(self, f:torch.Tensor) -> torch.Tensor:
        batch_size, n_vertices, channels = f.shape
        
        target = f[:, :, None, :].expand(
            batch_size,
            n_vertices,
            n_vertices,
            channels
        )
        
        source = f[:, None,:, :].expand(
            batch_size,
            n_vertices,
            n_vertices,
            channels
        )
        
        pair_features = torch.cat(
            [target, source],
            dim = -1
        )
        
        message = self.phi(pair_features)
        
        message_source = message * source
        
        kernel = (self.kernel.heat_kernel_matrices())
        
        y =  torch.einsum(
            "rvu, u, bvuc, rco -> bvo",
            kernel,
            self.kernel.mass,
            message_source,
            self.kernel.coefficients
        )
        
        return y


class IntegralOperatorModel(nn.Module):
    def __init__(
        self,
        layers: list[IntegralOperator],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.activation = nn.LeakyReLU()

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        f = f0

        for layer_index, layer in enumerate(self.layers):
            f = layer(f)

            if layer_index < len(self.layers) - 1:
                f = self.activation(f)

        return f