import torch
from torch import nn
from tqdm import tqdm


class DDPMPipeline(nn.Module):
    def __init__(self, beta_start=1e-4, beta_end=1e-2, n_timesteps=1000):
        super(DDPMPipeline, self).__init__()

        self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1 - self.betas

        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

        self.n_timesteps = n_timesteps

    def forward(self, x, t):
        eta = torch.randn_like(x).to(x.device)

        alpha_hat = self.alphas_hat.to(x.device)[t]

        return alpha_hat.sqrt() * x + (1 - alpha_hat).sqrt() * eta, eta

    def backward(self, model, noisy_x, t):
        noise_hat = model(noisy_x, t)
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

        return images, x
