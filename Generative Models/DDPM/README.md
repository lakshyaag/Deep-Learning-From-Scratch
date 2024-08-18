# Denoising Diffusion Probabilistic Models (DDPM)

This repository contains an implementation of Denoising Diffusion Probabilistic Models (DDPM) based on the paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239).

## Project Structure

- **data.py**: Contains a custom dataset class for loading images.
- **config.py**: Configuration file with hyperparameters and model settings.
- **models.py**: Contains the definition of the UNet model and the building blocks.
- **ddpm.py**: Contains the implementation of the DDPM scheduler.
- **trainer.py**: Script for training the DDPM model.
- **inference.py**: Script for running inference with the trained DDPM model.
- **utils.py**: Utility functions for image processing and noise scheduling.

## Usage

### Configuration

The `config.py` file contains hyperparameters and model settings that can be modified to suit the requirements of the experiment.

### Training

To train the model, run the `trainer.py` script:

```bash
python trainer.py
```

### Inference

The inference script allows for generating samples from the trained model.

To run inference with the trained model, use the `inference.py` script:

```bash
python inference.py /path/to/weights.pt --seed 42 --n 8 --sampler ddim --steps 50 --eta 0.0
```

## References

- [mattroz/diffusion-ddpm](https://github.com/mattroz/diffusion-ddpm)
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (Song et al., 2020)](https://arxiv.org/pdf/2010.02502)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/pdf/2102.09672)
