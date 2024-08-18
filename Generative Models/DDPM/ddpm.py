import torch
from torch import nn
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from models import Unet
from utils import linear_schedule, cosine_schedule


class DDPMPipeline(nn.Module):
    """
    Implements the Denoising Diffusion Probabilistic Model (DDPM) pipeline.

    This class handles the forward noising process, the backward denoising process, and the sampling process using both DDPM and DDIM.

    Args:
    - beta_start (float): The starting value of the beta schedule. Defaults to 1e-4.
    - beta_end (float): The ending value of the beta schedule. Defaults to 1e-2.
    - n_timesteps (int): The number of timesteps for the diffusion process. Defaults to 1000.
    - noise_schedule (str): The noise schedule to use. Can be either "linear" or "cosine".

    References:
    - Denoising Diffusion Probabilistic Models (Ho et al., 2020): https://arxiv.org/abs/2006.11239
    - Denoising Diffusion Implicit Models (Song et al., 2020): https://arxiv.org/pdf/2010.02502
    - Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021): https://arxiv.org/pdf/2102.09672
    """

    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
        n_timesteps: int = 1000,
        noise_schedule: str = "linear",
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

    def forward(self, x: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the forward noising process to the input tensor, as given in Equation 4 of the DDPM paper.

        Args:
        - x (torch.Tensor): The input tensor.
        - t (int): The current timestep.
        """
        eta = torch.randn_like(x).to(x.device)

        alpha_hat = self.alphas_hat.to(x.device)[t]

        return alpha_hat.sqrt() * x + (1 - alpha_hat).sqrt() * eta, eta

    def backward(
        self, model: Unet, noisy_x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the backward denoising process to the input tensor, predicting the noise tensor at timestep t.

        Args:
        - model (Unet): The denoising model.
        - noisy_x (torch.Tensor): The noised input tensor.
        - t (torch.Tensor): The current timestep.
        """
        noise_hat = model(noisy_x, t)
        return noise_hat

    @torch.no_grad()
    def sample_image(
        self, model: Unet, x_t: torch.Tensor, save_interval: int = 100
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Samples an image from the trained model using the DDPM sampling process, as given in Algorithm 2 of the DDPM paper.

        Args:
        - model (Unet): The denoising model.
        - x_t (torch.Tensor): The input tensor.
        - save_interval (int): The interval at which to save the generated images. Defaults to 100.
        """
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
    def sample_ddim(
        self,
        model: Unet,
        x_t: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
        save_interval: int = 100,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Samples an image from the trained model using the DDIM sampling process, as given in Equation 12 of the DDIM paper.

        Args:
        - model (nn.Module): The denoising model.
        - x_t (torch.Tensor): The input tensor.
        - ddim_steps (int): Number of sub-sequence steps to sample. Defaults to 50.
        - eta (float): The noise scale factor (DDIM when eta=0, DDPM when eta=1). Defaults to 0.0.
        - save_interval (int): The interval at which to save the generated images. Defaults to 100.
        """
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
                    else torch.tensor(1.0).to(x.device)  # alpha_0 = 1.0
                )

                # Equation 16 of the DDIM paper
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_hat_t_prev)
                    / (1 - alpha_hat_t)
                    * (1 - alpha_hat_t / alpha_hat_t_prev)
                )

                # Equation 12 of the DDIM paper
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
