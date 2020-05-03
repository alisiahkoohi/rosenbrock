import torch

class RosenbrockDistribution(object):
    r"""Implements Rosenbrock distribution

    Includes unnormalized density and analytic sampler.

    Based on https://arxiv.org/pdf/1903.09556.pdf

    Arguments:
        mu (torch.Tensor): the mean parameter, :math:`\mu \in \mathbb{R}`
        a (torch.Tensor): the precision parameter. :math:`\mu \in \mathbb{R}^{+}`
        b (torch.Tensor): Rosenbrock coefficients. :math:`\b_{i,j} \in \mathbb{R}^{+}`
        device (torch.Tensor): Device—e.g., 'cpu' or 'cuda'
    """
    def __init__(self, mu, a, b, device='cpu'):
        super(RosenbrockDistribution, self).__init__()

        assert a.shape == torch.Size([1]) and mu.shape == torch.Size([1]), \
            "a and mu need to be one dimensional Tensor—e.g., torch.Tensor([.5])"
        assert len(b.shape) == 2, \
            "b needs to be a two dimentional Tensor—e.g., torch.rand([2, 1])"
        assert a.data > 0.0 and (b.data > 0.0).all(), \
            "a and b need to be posititve"

        # This will output samples with dimension (n1-1)n2 + 1        
        self.n1, self.n2 = b.shape[0], b.shape[1]

        self.mu = mu
        self.a = a
        self.b = b
        self.device = device

        self.normal = torch.distributions.normal.Normal(0.0, 1.0)

    def nl_pdf(self, x):
        assert x.shape[1] == (self.n1-1)*self.n2 + 1,  "x has inconsistent size"
        nlpdf = self.a*(x[:, 0] - self.mu)**2
        for j in range(self.n2):
            for i in range(1, self.n1):
                nlpdf += self.b[i, j]*(x[:, i + j*(self.n1-1)] - x[:, i -1 + \
                    j*(self.n1-1)]**2)**2
        return nlpdf

    def sample(self, num_samples):
        x = torch.zeros([num_samples, (self.n1-1)*self.n2 + 1], device=self.device)
        for k in range(num_samples):
            x[k, 0] = self.normal.sample()/(2.0*self.a).sqrt() + self.mu
            for j in range(self.n2):
                for i in range(1, self.n1):
                    x[k, i + j*(self.n1-1)] = self.normal.sample()/(2.0*self.b[i, j]).sqrt() + \
                        x[k, i -1 + j*(self.n1-1)]**2
        return x
