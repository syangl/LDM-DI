import os, sys
# get the current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# get the project root directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(current_script_dir)))
# add the project root directory as the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_utils.config_util.attack_config import attack_args, PIPE_TYPE
from exp_utils.metric_util.evaluate import calculate_similarity
import torch
import torch.nn.functional as F
from tqdm import tqdm
from exp_utils.dataset_process_util.img_preprocess import get_embeded_prompt, pil2tensor


def execute_semansim(
    pipe,
    dataloader,
    choice="cos_sim",
):
    # Generate images
    prompt_embed_list = []
    dataset_images = []
    inversion_images = []
    for batch in tqdm(dataloader, desc="generating images...", total=len(dataloader)):
        index, prompt, input_image = batch['index'][0], batch['prompt'], batch['image']

        # save original image
        dataset_images.append(input_image.squeeze(0))

        # embedding prompt
        prompt_embeds, encode_prompt_addtions = get_embeded_prompt(pipe, prompt)
        prompt_embed_list.append(prompt_embeds.squeeze(0).cpu())

        tmp_forward_images = []
        for i in range(attack_args.num_samples):
            # get initial x_t latents
            generator = torch.Generator("cuda").manual_seed(attack_args.global_seed + i)
            # get freedom image
            image_output = pipe(
                prompt_embeds=prompt_embeds,
                attention_mask=encode_prompt_addtions.get('encoder_attention_mask', None),
                height=attack_args.generate_resolution,
                width=attack_args.generate_resolution,
                generator=generator,
                num_inference_steps=10,
                guidance_scale=7.5,
            ).images[0]

            tmp_forward_images.append(pil2tensor(image_output))

        inversion_images.append(tmp_forward_images)
    # calculate similarity
    metric_scores = calculate_similarity(inversion_images,
                                         dataset_images,
                                         choice=choice,
                                         prompt_embed_list=prompt_embed_list)

    return metric_scores
