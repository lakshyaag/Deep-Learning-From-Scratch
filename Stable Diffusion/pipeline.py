import torch
from clip import CLIP
from ddpm import DDPMSampler
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512

LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


def rescale(x, input_range, output_range, clamp=False):
    input_min, input_max = input_range
    output_min, output_max = output_range

    x -= input_min
    x *= (output_max - output_min) / (input_max - input_min)
    x += output_max

    if clamp:
        x = x.clamp(output_min, output_max)

    return x


def get_time_embedding(timestep):
    # (160, )
    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)

    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    # (1, 160) -> (1, 320)
    x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    return x


def generate(
    prompt,
    negative_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        # Initialize inference pipeline parameters
        if not (0 <= strength < 1):
            raise ValueError("Strength must be between 0 and 1")

        def to_idle(x):
            return x.to(idle_device) if idle_device is not None else x

        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()

        else:
            generator.manual_seed(seed)

        # Convert input prompt to embeddings using CLIP
        clip: CLIP = models["clip"]
        clip.to(device)

        conditional_tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77
        ).input_ids

        # B S
        conditional_tokens = torch.tensor(
            conditional_tokens, dtype=torch.long, device=device
        )

        # B S -> B S E
        condtional_context = clip(conditional_tokens)

        if do_cfg:
            # If classifier-free guidance is enabled
            # Convert negative prompt to embeddings using CLIP and concatenate with original prompt embeddings

            unconditional_tokens = tokenizer.batch_encode_plus(
                [negative_prompt], padding="max_length", max_length=77
            ).input_ids

            # B S
            unconditional_tokens = torch.tensor(
                unconditional_tokens, dtype=torch.long, device=device
            )

            # B S -> B S E
            unconditional_context = clip(unconditional_tokens)

            # B S E -> 2*B S E
            context = torch.cat([condtional_context, unconditional_context], dim=0)

        else:
            # B S E -> B S E
            context = condtional_context

        # Move CLIP back to idle device
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} not found")

        # B 4 H/8 W/8
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        # ---------------------------------------------------- #
        # TBD: img2img
        # ---------------------------------------------------- #

        # ---------------------------------------------------- #
        # txt2img
        # ---------------------------------------------------- #

        # Start with random noise
        latents = torch.randn(latents_shape, device=device, generator=generator)

        diffusion = models["diffusion"]
        diffusion.to(device)

        # Run generation loop
        for i, timestep in enumerate(tqdm(sampler.timesteps)):
            # 1 -> 1, 320
            time_embedding = get_time_embedding(timestep).to(device)

            # B 4 H/8 W/8
            model_input = latents

            if do_cfg:
                # B 4 H/8 W/8 -> 2*B 4 H/8 W/8
                model_input = model_input.repeat(2, 1, 1, 1)

            # Get predicted noise
            # B 4 H/8 W/8, B S E, 1 320 -> B 4 H/8 W/8
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # 2*B 4 H/8 W/8 -> B 4 H/8 W/8, B 4 H/8 W/8
                output_cond, output_uncond = model_output.chunk(2, dim=0)
                model_output = (output_cond - output_uncond) * cfg_scale + output_uncond

            # Remove predicted noise from the latent tensor
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # Pass generated latent tensor through the decoder to get the final image
        decoder = models["decoder"]
        decoder.to(device)

        # B 4 H/8 W/8 -> B 3 H W
        generated_image = decoder(latents)
        to_idle(decoder)

        # Reverse normalization
        generated_image = rescale(generated_image, (-1, 1), (0, 255), clamp=True)

        # B 3 H W -> B H W 3
        generated_image = (
            generated_image.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        )
        return generated_image[0]
