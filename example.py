import torch
import numpy as np
from  rosenbrock import rosenbrock
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':

    # Initialize parameters
    mu = torch.Tensor([0.0], device=device)
    a = torch.Tensor([.5], device=device)
    b = torch.ones([2, 1], device=device)

    # Define the distribution
    rosen_dist = rosenbrock.RosenbrockDistribution(mu, a, b, device='cpu')

    # Sample the distribution
    samples = rosen_dist.sample(int(5e3))

    # Compute negative-log of the density function for the samples
    nl_pdf = rosen_dist.nl_pdf(samples)

    samples = samples.cpu().numpy()
    fig = plt.figure(dpi=150, figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=.5, color="#db76bf")
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    plt.grid()
    plt.title(r"$\mathbf{x} \sim p_X(\mathbf{x})$")
    plt.show()
