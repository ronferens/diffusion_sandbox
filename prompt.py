import numpy as np

prompt_att_gender = ['man', 'woman']
prompt_att_look = ['forward', 'away', 'to the side']
prompt_att_age = ['6', '10', '14', '16']

prompt_att_image_color = 'a color frontal photo'

prompt_att_covered_at = ['mouth', 'chin']
prompt_att_covered_by = ['his fingers', 'palms']
prompt_att_glasses_color = ['semi dark', 'dark']
prompt_att_glasses_rim = ['halfrim', 'rimless']
prompt_att_mask_type = ['facial', 'surgical']
prompt_att_mask_color = ['white', 'blue', 'black', 'green', 'color']
prompt_att_eating = ['taking a bite of food', 'take a sip from cup', 'take a sip from bottle']
prompt_att_phone_usage = ['talking on cellphone']


def create_prompt_att(prompt_att):
    return prompt_att[np.random.randint(low=0, high=len(prompt_att), size=1, dtype=np.uint)[0]]


face_occluded = f'{prompt_att_image_color} of a {create_prompt_att(prompt_att_gender)},' \
                f'looking {create_prompt_att(prompt_att_look)},' \
                f'covering {create_prompt_att(prompt_att_covered_at)} with {create_prompt_att(prompt_att_covered_by)}'

face_occluded_and_sunglasses = f'{prompt_att_image_color} of a {create_prompt_att(prompt_att_gender)}, looking {create_prompt_att(prompt_att_look)}, ' \
                               f'covering {create_prompt_att(prompt_att_covered_at)} with {create_prompt_att(prompt_att_covered_by)},' \
                               f'wearing {create_prompt_att(prompt_att_glasses_color)} {create_prompt_att(prompt_att_glasses_rim)} sunglasses'

face_occluded_and_glasses = f'{prompt_att_image_color} of a {create_prompt_att(prompt_att_gender)}, looking {create_prompt_att(prompt_att_look)}, ' \
                            f'covering {create_prompt_att(prompt_att_covered_at)} with {create_prompt_att(prompt_att_covered_by)}, ' \
                            f'wearing {create_prompt_att(prompt_att_glasses_rim)} transparent glasses'

mask = f'{prompt_att_image_color} of a {create_prompt_att(prompt_att_gender)}, looking {create_prompt_att(prompt_att_look)}, ' \
       f'wearing {create_prompt_att(prompt_att_mask_color)} {create_prompt_att(prompt_att_mask_type)} mask'

mask_and_glasses = f'{prompt_att_image_color} of a {create_prompt_att(prompt_att_gender)}, looking {create_prompt_att(prompt_att_look)}, ' \
                   f'wearing {create_prompt_att(prompt_att_glasses_rim)} transparent glasses' \
                   f'and {create_prompt_att(prompt_att_mask_color)} {create_prompt_att(prompt_att_mask_type)} mask'

mask_and_sunglasses = f'{prompt_att_image_color} of a {create_prompt_att(prompt_att_gender)}, looking {create_prompt_att(prompt_att_look)}, ' \
                      f'wearing {create_prompt_att(prompt_att_glasses_color)} {create_prompt_att(prompt_att_glasses_rim)} sunglasses' \
                      f'and {create_prompt_att(prompt_att_mask_color)} {create_prompt_att(prompt_att_mask_type)} mask'

smoking = f'{prompt_att_image_color} of a {create_prompt_att(prompt_att_gender)}, looking {create_prompt_att(prompt_att_look)}, ' \
          'inhale the cigarette smoke into the lungs'

eating = f'{prompt_att_image_color} of a {create_prompt_att(prompt_att_gender)}, looking {create_prompt_att(prompt_att_look)}, ' \
         'taking a bite of food'

kids = f'{prompt_att_image_color} of a child, of age {create_prompt_att(prompt_att_age)}, looking {create_prompt_att(prompt_att_look)} ' \
       f'holding a cellphone'

prompts = []


def prompt_creation(batch_size: int):
    for _ in range(batch_size):
        rand_prompt = kids
        rand_prompt = 'sitting in front seat of the car in parking lot'
        rand_prompt += f'50mm portrait photography, daylighting photography--beta --ar 3:2  --beta --upbeta'

        prompts.append(rand_prompt)

    return prompts
