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


def main():
    os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)
    seed = 41
    model = Unet(
        image_size=cfg.IMAGE_SIZE,
        in_channels=cfg.IN_CHANNELS,
        base_channels=cfg.BASE_CHANNELS,
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS,
        n_groups=cfg.N_GROUPS,
    ).to(device)

    pipeline = DDPMPipeline(n_timesteps=cfg.N_TIMESTEPS).to(device)

    checkpoint = torch.load(
        f"{cfg.MODEL_DIR}/9p4viyr6_ddpm_checkpoint_final.pt", map_location=device
    )

    model.load_state_dict(checkpoint["model"])
    pipeline.load_state_dict(checkpoint["pipeline"])
    model.eval()

    torch.manual_seed(seed)
    noise = torch.randn(
        cfg.EVAL_BATCH_SIZE, cfg.IN_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE
    ).to(device)

    images, x_0 = pipeline.sample_image(model, noise, save_interval=cfg.SAVE_INTERVAL)

    fig = plt.figure()
    ims = []

    for i in range(0, len(images)):
        image_grid = reverse_transform(make_grid(images[i], nrow=4, padding=4))
        im = plt.imshow(image_grid, animated=True)
        ims.append([im])

    plt.axis("off")
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat=False)

    ani.save(f"{cfg.SAMPLE_DIR}/ddpm_sample_{seed}.gif", writer="pillow")
    print("Saved animation!")

    x_0 = reverse_transform(make_grid(x_0, nrow=4, padding=4))
    plt.imshow(x_0)
    plt.axis("off")
    plt.savefig(f"{cfg.SAMPLE_DIR}/ddpm_sample_{seed}.png")
    plt.close()
    print("Saved image!")


if __name__ == "__main__":
    main()
