"""
References:
* https://huggingface.co/docs/diffusers/stable_diffusion
"""
from diffusers import DiffusionPipeline
from diffusers import DPMSolverSDEScheduler
from diffusers import AutoencoderKL
from diffusers.utils import make_image_grid
import torch
from utils import set_inputs, save_generation_metadata
from typing import List
from datetime import datetime
from os.path import exists, join
from os import mkdir
import numpy as np
from enum import Enum


DEFAULT_NUM_OF_DISP_NCOLS = 4
OUTPUT_DIR = 'output'


class ProcessType(Enum):
    BATCH = 0,
    ITERATIVE = 1


def main():
    # Setting the processing type
    process_type = ProcessType.ITERATIVE

    # Setting the device to run on
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    if not exists(OUTPUT_DIR):
        mkdir(OUTPUT_DIR)

    # Ramping up the Diffusion model pipeline
    # diffusion_model = 'runwayml/stable-diffusion-v1-5'
    diffusion_model = 'stabilityai/stable-diffusion-xl-base-1.0'

    pipe = DiffusionPipeline.from_pretrained(diffusion_model,
                                             torch_dtype=torch.float16,
                                             variant="fp16",
                                             use_safetensors=True)

    # # Setting the model's diffusion scheduler
    # pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
    #
    # # Setting the model's autoencoder
    # vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse', torch_dtype=torch.float16).to(device)
    # pipe.vae = vae

    pipe.enable_attention_slicing()
    pipe.to(device)

    # Setting the prompt
    batch_size = 4
    # prompt = "portrait photo of a old warrior chief woman"
    # prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
    # prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"

    prompt = "a color image of a driver smoking cigarette looking to the camera"
    prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"

    # prompt = [
    #     "portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    #     "portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    #     "portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    #     "portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    # ]

    if isinstance(prompt, List):
        batch_size = len(prompt)

    negative_prompt = batch_size * ["disfigured, ugly, bad, immature, cartoon, anime, painting"]

    # Setting the generator's seed
    inputs, seeds_list = set_inputs(prompt=prompt,
                                    batch_size=batch_size,
                                    seed=None,
                                    nsteps=25)

    if process_type == ProcessType.ITERATIVE:
        images = []
        for idx in range(batch_size):
            # Generating the images one by one
            img = pipe(
                prompt=inputs['prompt'][idx],
                generator=inputs['generator'][idx],
                num_inference_steps=inputs['num_inference_steps'],
                negative_prompt=negative_prompt[idx],
            ).images[0]
            images.append(img)

    elif process_type == ProcessType.BATCH:
        # Generating the images for all given prompts
        images = pipe(
            **inputs,
            negative_prompt=negative_prompt,
        ).images
    else:
        raise ValueError('Unsupported processing type')

    grid_nrows = max(batch_size // DEFAULT_NUM_OF_DISP_NCOLS, 1)
    grid_ncols = batch_size if batch_size < DEFAULT_NUM_OF_DISP_NCOLS else DEFAULT_NUM_OF_DISP_NCOLS
    grid = make_image_grid(images, grid_nrows, grid_ncols)
    grid.show()

    # Saving the prompts and their seed
    output_filename = datetime.now().strftime('%d%m%Y_%H%M%S')
    grid.save(join(OUTPUT_DIR, output_filename + '.png'))
    save_generation_metadata(inputs, seeds_list, join(OUTPUT_DIR, output_filename))


if __name__ == '__main__':
    main()
