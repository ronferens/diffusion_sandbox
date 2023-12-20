import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
import os

torch._dynamo.config.suppress_errors = True

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe = pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

filename = "/home/blink/Pictures/20230527-060846-7FD03E02_5_size100.jpg"
image = Image.open(filename)
org_h, org_w = image.size
init_image = image.resize((1024, 1024))

# init_image = load_image(filename).convert("RGB")
prompt = "a young woman with white shirt looking forward"
output_img = pipe(prompt, image=init_image, strength=0.5, guidance_scale=7.5).images[0]
output_img = output_img.resize((org_h, org_w))
output_img.save("img2img.jpg")
