import torch
import torch.nn as nn
from tqdm import tqdm
from models import Diffusion


class DDPMPipeline(nn.Module):
    def __init__(
        self,
        generator: torch.Generator,
        beta_start=0.00085,
        beta_end=0.0120,
        n_timesteps=1000,
    ):
        super(DDPMPipeline, self).__init__()

        self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.n_timesteps = n_timesteps
        self.generator = generator

    def forward(self, x, t):
        eta = torch.randn(
            x.shape, device=x.device, generator=self.generator, dtype=x.dtype
        )

        alpha_hat = self.alphas_hat.to(x.device)[t]

        return alpha_hat.sqrt() * x + (1 - alpha_hat).sqrt() * eta, eta

    def backward(self, model: Diffusion, noisy_x, y, t):
        noise_hat = model(noisy_x, y, t)
        return noise_hat

    @torch.no_grad()
    def sample_image(self, model, x_t):
        x = x_t
        images = []

        for t in tqdm(range(self.n_timesteps - 1, -1, -1)):
            ts = t * torch.ones(x.shape[0], dtype=torch.long, device=x.device)
            noise_hat = model(x, ts)

            beta_t = self.betas.to(x.device)[t]
            alpha_t = self.alphas.to(x.device)[t]
            alpha_hat_t = self.alphas_hat.to(x.device)[t]

            alpha_hat_t_prev = self.alphas_hat.to(x.device)[t - 1]
            beta_hat_t = (1 - alpha_hat_t_prev) / (1 - alpha_hat_t) * beta_t
            variance = torch.sqrt(beta_hat_t) * torch.randn_like(x) if t > 0 else 0

            x = (
                torch.pow(alpha_t, -0.5)
                * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * noise_hat))
                + variance
            )

            images.append(x.cpu())

        return images
