"""
References:
* https://huggingface.co/docs/diffusers/stable_diffusion
"""
import os
from diffusers import DiffusionPipeline
import torch
from utils import set_inputs, save_generation_metadata
from typing import List
from datetime import datetime
from os.path import join, exists
import numpy as np
from enum import Enum
from transformers import CLIPProcessor, CLIPModel
from multiprocessing import Pool
import json
import prompt as prompts

NUM_OF_GPU = 4

DEFAULT_NUM_OF_DISP_NCOLS = 4
OUTPUT_DIR = r'D:\diff_models\test_test_json2'
OUTPUT_DIR_DEFECT = r'defects'
class ClipRunner:
    CLASSES_WITH_DESCRIPTION = {"child": "a picture of a child looking forward",
                                "child_talking_on_phone": "a picture of a child holding a cellphone"}
    # CLASSES_WITH_DESCRIPTION = {"talking on phone": "a person is talking on the phone",
    #                             "normal": "a person is walking without holding nothing on her hands"}

    def __init__(self, device=None):
        if device is None:
            device = f"cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                if torch.backends.mps.is_available():
                    device = "mps"
                    # if torch.cuda.is_available():
                    #     device = f'cuda:{device}'
        self._model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        cls_names, cls_desc = zip(*self.CLASSES_WITH_DESCRIPTION.items())
        self._cls_names = cls_names
        self._cls_desc = cls_desc
        self._device = device

    @torch.no_grad()
    def run(self, img: np.ndarray):
        model_input = self._processor(text=self._cls_desc, images=img, return_tensors="pt", padding=True).to(
            self._device)
        outputs = self._model(**model_input)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(
            dim=1).flatten().cpu().numpy()  # we can take the softmax to get the label probabilities
        return {k: v for k, v in zip(self._cls_names, probs)}


def run_clip_verification(clip_runner, img):
    preds = clip_runner.run(img)
    class_names, confidence_scores = zip(*preds.items())
    confidence_scores = np.asarray(confidence_scores)

    max_class, max_conf = max(zip(class_names, confidence_scores), key=lambda x: x[1])

    return max_class == "child_talking_on_phone"

class ProcessType(Enum):
    BATCH = 0,
    ITERATIVE = 1


def run_inference(device_id: int):
    # Setting the processing type
    process_type = ProcessType.ITERATIVE
    batch_size = 2
    # Setting the device to run on
    if torch.cuda.is_available():
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ramping up the Diffusion model pipeline
    diffusion_model = 'stabilityai/stable-diffusion-xl-base-1.0' #'runwayml/stable-diffusion-v1-5'

    pipe = DiffusionPipeline.from_pretrained(diffusion_model,
                                             torch_dtype=torch.float16,
                                             variant="fp16",
                                             use_safetensors=True)

    pipe.enable_attention_slicing()
    pipe.to(device)

    # Setting the prompt

    # JSON = r'D:\diff_models\eating\25122023_062156.json' #D:\diff_models\test_test\28122023_092832.json
    JSON = r'D:\diff_models\test_test\20240104_095030.json'# r'D:\diff_models\test_test\20240104_095030.json'

    # Setting the prompt
    if os.path.isfile(JSON):
        data = json.load(open(JSON))
        # Extract and save keys to lists
        prompt_from_json, num_inference_steps, seeds_from_json = data['prompt'], data['num_inference_steps'], data['seeds']

        row_ind = np.arange(device_id, len(prompt_from_json), NUM_OF_GPU)
        # prompt creation
        prompt = [prompt_from_json[ind] for ind in row_ind]
        seeds = [seeds_from_json[ind] for ind in row_ind]

        batch_size = len(prompt)
    else:
        prompt = prompts.prompt_creation(batch_size=batch_size)
        seeds = None
        if isinstance(prompt, List):
            batch_size = len(prompt)

    negative_prompt = batch_size * ["disfigured, ugly, bad, immature, cartoon, anime, painting"]


    # Setting the generator's seed
    inputs, seeds_list = set_inputs(prompt=prompt,
                                    batch_size=batch_size,
                                    seed=seeds,
                                    nsteps=20)

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

    clip_runner = ClipRunner()

    # output_filename = datetime.now().strftime('%d%m%Y_%H%M%S')
    pid = os.getpid()
    for idx, img in enumerate(images):
        output_filename = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'gen_img_{output_filename}_{pid}_{idx}.jpg'
        res = run_clip_verification(clip_runner, img)
        if res:
            target_path = join(OUTPUT_DIR, output_filename)
        else:
            if not exists(join(OUTPUT_DIR, OUTPUT_DIR_DEFECT)):
                os.makedirs(join(OUTPUT_DIR, OUTPUT_DIR_DEFECT), exist_ok=True)
            target_path = join(OUTPUT_DIR, OUTPUT_DIR_DEFECT, output_filename)
        img.save(target_path)


    # Saving the prompts and their seed
    output_filename = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_generation_metadata(inputs, seeds_list, join(OUTPUT_DIR, output_filename))


def main():
    devices = [0, 1, 2, 3]

    with Pool(5) as p:
        print(p.map(run_inference, devices))


if __name__ == '__main__':
    main()
