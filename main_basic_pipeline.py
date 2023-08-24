from diffusers import DDPMPipeline
from diffusers import DDPMScheduler, UNet2DModel
import torch
from PIL import Image
from plot_utils import cvt_img_for_display, animate_sampling
from tqdm import tqdm


def main():
    # Setting the device to run on
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # =============================================
    # Running the full integrated pipeline
    # =============================================
    # ddpm = DDPMPipeline.from_pretrained('google/ddpm-cat-256', use_safetensors=True).to(device)
    # img = ddpm(num_inference_steps=25).images[0]
    # img.show()

    # =============================================
    # Deconstruct the DDPM pipeline
    # =============================================
    scheduler = DDPMScheduler.from_pretrained('google/ddpm-cat-256')
    model = UNet2DModel.from_pretrained('google/ddpm-cat-256', use_safetensors=True).to(device)

    # Setting the scheduler's number of timesteps to run the denoising process
    scheduler.set_timesteps(50)
    print(scheduler.timesteps)

    # Creating a random noise with the same shape as the desired output
    sample_size = model.config.sample_size
    noise = torch.randn((1, 3, sample_size, sample_size)).to(device)
    print(f'noise shape: {noise.shape}')

    # The denoising loop that iterates over the timesteps
    input = noise
    intermediate_imgs = []

    for t in tqdm(scheduler.timesteps, desc='Running denoising loop'):
        intermediate_imgs.append(input)
        with torch.no_grad():
            noise_residual = model(input, t).sample
        previous_noisy_sample = scheduler.step(noise_residual, t, input).prev_sample
        input = previous_noisy_sample

    animate_sampling(intermediate_imgs)

    img = cvt_img_for_display(input)
    img = Image.fromarray(img)
    img.show()


if __name__ == '__main__':
    main()