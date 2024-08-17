import torch
import wandb
from config import Config
from ddpm import DDPMPipeline
from models import Unet
from rich import print
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from utils import get_transforms, reverse_transform

wandb.require("core")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()


def load_data(data_dir):
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(cfg.IMAGE_SIZE),
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
    )

    return train_loader


def train_epoch(model, pipeline, optimizer, lr_scheduler, loss_fn, train_loader):
    model.train()
    train_loss = 0.0

    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        t = torch.randint(0, cfg.N_TIMESTEPS, size=(1,)).to(device).long()

        noisy_x, eta = pipeline(images, t)
        noisy_x = noisy_x.to(device)

        noise_hat = pipeline.backward(model, noisy_x, t)
        loss = loss_fn(noise_hat, eta)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    wandb.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]})

    return train_loss


def eval_epoch(model: Unet, pipeline: DDPMPipeline, epoch, config: Config):
    model.eval()

    noise = torch.randn(
        config.EVAL_BATCH_SIZE, config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE
    ).to(device)

    _, x_0 = pipeline.sample_image(model, noise)

    return x_0


def main():
    train_loader = load_data(cfg.DATA_DIR)

    pipeline = DDPMPipeline(n_timesteps=cfg.N_TIMESTEPS).to(device)
    model = Unet(
        image_size=cfg.IMAGE_SIZE,
        in_channels=cfg.IN_CHANNELS,
        base_channels=cfg.BASE_CHANNELS,
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS,
        n_groups=cfg.N_GROUPS,
    ).to(device)

    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameter count: {total_param_count / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=len(train_loader) * cfg.N_EPOCHS,
        last_epoch=-1,
        eta_min=1e-9,
    )

    loss_fn = nn.MSELoss()

    run = wandb.init(
        project=cfg.EXPERIMENT_NAME,
        config={
            **cfg.__dict__,
            "total_param_count": total_param_count,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": lr_scheduler.__class__.__name__,
            "loss_fn": loss_fn.__class__.__name__,
        },
    )

    print(f"Starting DDPM training with config: {cfg}")

    for epoch in range(cfg.N_EPOCHS):
        train_loss = train_epoch(
            model, pipeline, optimizer, lr_scheduler, loss_fn, train_loader
        )

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.3e}"
        )

        if epoch % cfg.EVAL_EVERY_EPOCH == 0 or epoch == cfg.N_EPOCHS - 1:
            x_0 = eval_epoch(model, pipeline, epoch, cfg)
            wandb.log(
                {
                    "generated_images": [
                        wandb.Image(
                            reverse_transform(make_grid(x_0, nrow=4, padding=4))
                        )
                    ]
                }
            )

        if epoch % cfg.CHECKPOINT_EVERY_EPOCH == 0 or epoch == cfg.N_EPOCHS - 1:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "config": cfg,
                "pipeline": pipeline.state_dict(),
            }

            torch.save(
                checkpoint, f"{cfg.MODEL_DIR}/{run.id}_ddpm_checkpoint_{epoch}.pt"
            )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "config": cfg,
        "pipeline": pipeline.state_dict(),
    }

    torch.save(checkpoint, f"{cfg.MODEL_DIR}/{run.id}_ddpm_checkpoint_final.pt")
    print(f"Model training completed, last checkpoint saved at {cfg.MODEL_DIR}")


if __name__ == "__main__":
    main()
