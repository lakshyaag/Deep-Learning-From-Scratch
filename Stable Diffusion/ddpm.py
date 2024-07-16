import torch
import numpy as np


class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        n_training_steps: int = 1000,
        min_beta: float = 0.00085,
        max_beta: float = 0.0120,
    ):
        self.betas = (
            torch.linspace(
                min_beta**0.5, max_beta**0.5, n_training_steps, dtype=torch.float32
            )
            ** 2
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.ones = torch.tensor(1.0)

        self.n_training_steps = n_training_steps
        self.generator = generator
        self.timesteps = torch.from_numpy(np.arange(0, n_training_steps)[::-1].copy())

    def set_inference_timesteps(self, n_inference_steps: int = 50):
        self.n_inference_steps = n_inference_steps

        step_ratio = self.n_training_steps // self.n_inference_steps

        # Define the timesteps to use for inference
        timesteps = (
            (np.arange(0, n_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.n_training_steps // self.n_inference_steps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_bar_t = self.alpha_bars[timestep]
        alpha_bar_prev_t = self.alpha_bars[prev_t] if prev_t >= 0 else self.ones

        current_beta_t = 1 - alpha_bar_t / alpha_bar_prev_t

        variance = (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * current_beta_t

        variance = variance.clamp(min=1e-20)

        return variance

    def set_strength(self, strength=1):
        """
        Set how much noise to add to the input image.
        More noise (strength ~ 1) means that the output will be further from the input image.
        Less noise (strength ~ 0) means that the output will be closer to the input image.
        """

        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def add_noise(
        self, x: torch.FloatTensor, timestep: torch.IntTensor
    ) -> torch.FloatTensor:
        """
        Add noise to an input tensor x at a given timestep t
        """

        alpha_bars = self.alpha_bars.to(device=x.device, dtype=x.dtype)
        timestep = timestep.to(x.device)

        sqrt_alpha_bars = alpha_bars[timestep] ** 0.5
        sqrt_alpha_bars = sqrt_alpha_bars.flatten()

        # Repeat the sqrt_alpha_bars tensor to match the input tensor shape
        while len(sqrt_alpha_bars.shape) < len(x.shape):
            sqrt_alpha_bars = sqrt_alpha_bars.unsqueeze(-1)

        sqrt_one_minus_alpha_bars = (self.ones - alpha_bars[timestep]) ** 0.5
        sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.flatten()

        # Repeat the sqrt_one_minus_alpha_bars tensor to match the input tensor shape
        while len(sqrt_one_minus_alpha_bars.shape) < len(x.shape):
            sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.unsqueeze(-1)

        noise = torch.randn(
            x.shape, generator=self.generator, device=x.device, dtype=x.dtype
        )

        mean = sqrt_alpha_bars * x
        std_dev = sqrt_one_minus_alpha_bars

        noisy_samples = mean + std_dev * noise

        return noisy_samples

    def step(self, timestep: int, latent: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_prev_t = self.alpha_bars[prev_t] if prev_t >= 0 else self.ones

        beta_t = 1 - alpha_bar_t
        beta_prev_t = 1 - alpha_bar_prev_t

        # Compute the predicted original sample
        x0_pred = (latent - beta_t**0.5) * model_output * (1 / alpha_bar_t**0.5)

        # Compute mean and variance of the posterior
        mean = ((alpha_bar_prev_t**0.5 * beta_t) / beta_t * x0_pred) + (
            (alpha_bar_t**0.5 * beta_prev_t) / beta_t
        ) * latent

        std_dev = 0

        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )

            std_dev = self._get_variance(t) ** 0.5 * noise

        mean = mean + std_dev

        return mean
