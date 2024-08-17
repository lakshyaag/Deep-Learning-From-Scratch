import torch
from torch import nn
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from utils import linear_schedule, cosine_schedule


class DDPMPipeline(nn.Module):
    def __init__(
        self, beta_start=1e-4, beta_end=1e-2, n_timesteps=1000, noise_schedule="linear"
    ):
        super(DDPMPipeline, self).__init__()

        self.betas = (
            linear_schedule(beta_start, beta_end, n_timesteps)
            if noise_schedule == "linear"
            else cosine_schedule(n_timesteps)
        )
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
    def sample_image(self, model, x_t, save_interval=100):
        x = x_t
        images = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            inference_task = progress.add_task(
                "[purple]Generating (DDPM)...", total=self.n_timesteps
            )

            for t in range(self.n_timesteps - 1, -1, -1):
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

                if save_interval != 0 and (t % save_interval == 0 or t == 0):
                    images.append(x.cpu())

                progress.update(
                    inference_task,
                    advance=1,
                    description="[purple]Generating (DDPM)...",
                )

        return images, x

    @torch.no_grad()
    def sample_ddim(self, model, x_t, ddim_steps=50, eta=0.0, save_interval=100):
        x = x_t
        images = []
        step_size = self.n_timesteps // ddim_steps
        save_every_n_steps = self.n_timesteps // save_interval

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            inference_task = progress.add_task(
                "[purple]Generating (DDIM)...", total=ddim_steps
            )

            for t in range(self.n_timesteps - 1, -1, -step_size):
                prev_t = t - step_size

                ts = t * torch.ones(x.shape[0], dtype=torch.long, device=x.device)
                noise_hat = model(x, ts)

                alpha_hat_t = self.alphas_hat.to(x.device)[t]

                alpha_hat_t_prev = (
                    self.alphas_hat.to(x.device)[prev_t]
                    if prev_t >= 0
                    else torch.tensor(1.0).to(x.device)
                )

                sigma_t = eta * torch.sqrt(
                    (1 - alpha_hat_t_prev)
                    / (1 - alpha_hat_t)
                    * (1 - alpha_hat_t / alpha_hat_t_prev)
                )

                pred_x_0 = (x - (1 - alpha_hat_t) ** 0.5 * noise_hat) / alpha_hat_t**0.5

                direction_x_t = (1 - alpha_hat_t_prev - sigma_t**2) ** 0.5 * noise_hat

                x = (
                    alpha_hat_t_prev**0.5 * pred_x_0
                    + direction_x_t
                    + sigma_t * torch.randn_like(x)
                )

                if save_every_n_steps != 0 and (
                    t // step_size % save_every_n_steps == 0
                ):
                    images.append(x.cpu())

                progress.update(
                    inference_task,
                    advance=1,
                    description="[purple]Generating (DDIM)...",
                )

        return images, x
