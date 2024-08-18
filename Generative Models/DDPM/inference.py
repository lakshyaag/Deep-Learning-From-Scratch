import typer
import os
import torch
from rich import print
from config import Config
from ddpm import DDPMPipeline
from matplotlib import animation
from matplotlib import pyplot as plt
from models import Unet
from torchvision.utils import make_grid
from utils import reverse_transform

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    weights: str,
    sampler: str = "ddpm",
    seed: int = 42,
    n: int = cfg.EVAL_BATCH_SIZE,
    steps: int = cfg.N_TIMESTEPS,
    eta: float = 0.0,
):
    """
    Generate samples from a trained DDPM model.

    Args:\n
    - weights (str): The path to the trained model weights.\n
    - sampler (str): The sampler to use. Either `ddpm` or `ddim`. Defaults to `ddpm`.\n
    - seed (int): The random seed to use. Defaults to 42.\n
    - n (int): The number of samples to generate. Defaults to 8.\n
    - steps (int): The number of steps to run DDIM for. Defaults to 1000.\n
    - eta (float): The noise level for DDIM. Defaults to 0.0.
    """
    os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)

    model = Unet(
        image_size=cfg.IMAGE_SIZE,
        in_channels=cfg.IN_CHANNELS,
        base_channels=cfg.BASE_CHANNELS,
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS,
        n_groups=cfg.N_GROUPS,
    ).to(device)

    pipeline = DDPMPipeline(
        n_timesteps=cfg.N_TIMESTEPS, noise_schedule=cfg.NOISE_SCHEDULE
    ).to(device)

    checkpoint = torch.load(weights, map_location=device)

    model.load_state_dict(checkpoint["model"])
    # pipeline.load_state_dict(checkpoint["pipeline"])
    model.eval()

    torch.manual_seed(seed)
    noise = torch.randn(n, cfg.IN_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(device)

    if sampler == "ddpm":
        images, x_0 = pipeline.sample_image(
            model, noise, save_interval=cfg.SAVE_INTERVAL
        )
    elif sampler == "ddim":
        images, x_0 = pipeline.sample_ddim(
            model,
            noise,
            ddim_steps=steps,
            eta=eta,
            save_interval=cfg.SAVE_INTERVAL,
        )

    # Save the images as a GIF
    fig = plt.figure()
    ims = []

    for i in range(0, len(images)):
        image_grid = reverse_transform(make_grid(images[i], nrow=4, padding=4))
        im = plt.imshow(image_grid, animated=True)
        ims.append([im])

    plt.axis("off")
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat=False)

    ani.save(f"{cfg.SAMPLE_DIR}/ddpm_sample_{seed}_{sampler}.gif", writer="pillow")
    print("Saved animation!")

    # Save the final image
    x_0 = reverse_transform(make_grid(x_0, nrow=4, padding=4))
    plt.imshow(x_0)
    plt.axis("off")
    plt.savefig(f"{cfg.SAMPLE_DIR}/ddpm_sample_{seed}_{sampler}.png")
    plt.close()
    print("Saved image!")


if __name__ == "__main__":
    typer.run(main)
