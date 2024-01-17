from typing import List, Dict, Union
import torch
import json
import numpy as np


def set_inputs(prompt: Union[str, List[str]], batch_size: int = 1, nsteps: int = 20, device: str = 'cuda',
               seed: int = None) -> Dict:
    # Incase a single prompt was given - setting the number of outputs according to the requested batch size
    if isinstance(prompt, str):
        prompt = batch_size * [prompt]

    if seed is None:
        # Applying random seed to each prompt
        seeds_list = np.random.randint(batch_size * 100, size=batch_size)
    elif isinstance(seed, int):
        # Applying the same seed for all prompts
        seeds_list = batch_size * [seed]
    else:
        # Applying the same seed for all prompts
        seeds_list = seed

    generator_list = []
    for idx in range(len(prompt)):
        generator_list.append(torch.Generator(device).manual_seed(int(seeds_list[idx])))
    num_inference_steps = nsteps

    return {'prompt': prompt, 'generator': generator_list, 'num_inference_steps': num_inference_steps}, seeds_list


def save_generation_metadata(inputs: Dict, seeds: List, out_filename: str) -> None:
    # Setting the parameters to save
    inputs.pop('generator')
    if isinstance(seeds, list):
        inputs['seeds'] = seeds
    else:
        inputs['seeds'] = seeds.tolist()

    inputs['num_inference_steps'] = len(inputs['seeds']) * [inputs['num_inference_steps']]

    # Serializing json
    json_object = json.dumps(inputs, indent=4)

    # Writing to sample.json
    with open(out_filename + '.json', 'w') as outfile:
        outfile.write(json_object)
