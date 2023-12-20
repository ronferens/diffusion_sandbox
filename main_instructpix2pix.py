import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
                                                              torch_dtype=torch.float16,
                                                              safety_checker=None)
pipe.enable_attention_slicing()
pipe.to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

filename = "/home/blink/Pictures/20230527-053948-731A1E22_5_size100.jpg"
image = PIL.Image.open(filename)
org_h, org_w = image.size
image = image.resize((org_h // 4, org_w // 4))

prompt = "make her look asian, wide eyes"
images = pipe(prompt, image=image, num_inference_steps=25, image_guidance_scale=1.2).images

out_img = images[0].resize((org_h, org_w))
out_img.save("glass_0111.jpg")