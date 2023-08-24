from typing import List, Dict, Union
import torch
import json


def set_inputs(prompt: Union[str, List[str]], batch_size: int = 1, nsteps: int = 20, device: str = 'cuda',
               seed: int = None, seed_offset: int = 0) -> Dict:
    # Incase a single prompt was given - setting the number of outputs according to the requested batch size
    if isinstance(prompt, str):
        prompt = batch_size * [prompt]

    generator_list = []
    seeds_list = []
    for idx in range(len(prompt)):
        seeds_list.append(seed if seed is not None else (idx + seed_offset))
        generator_list.append(torch.Generator(device).manual_seed(seeds_list[-1]))
    num_inference_steps = nsteps

    return {'prompt': prompt, 'generator': generator_list, 'num_inference_steps': num_inference_steps}, seeds_list


def save_generation_metadata(inputs: Dict, seeds: List, out_filename: str) -> None:
    # Setting the parameters to save
    inputs.pop('generator')
    inputs['seeds'] = seeds
    inputs['num_inference_steps'] = len(inputs['seeds']) * [inputs['num_inference_steps']]

    # Serializing json
    json_object = json.dumps(inputs, indent=4)

    # Writing to sample.json
    with open(out_filename + '.json', 'w') as outfile:
        outfile.write(json_object)
