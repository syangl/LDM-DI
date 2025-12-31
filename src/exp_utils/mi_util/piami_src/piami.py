import os, sys
# get the current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# get the project root directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(current_script_dir)))
# add the project root directory as the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import json
import shutil
from tqdm import tqdm
from exp_utils.config_util.attack_config import attack_args, get_unet_kwargs, PIPE_TYPE, TRAIN_OUTPUT_DIR, HOLD_OUT_OUTPUT_DIR, dataset_config_1, dataset_config_2, MI_DIR
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.metric_util.evaluate import evaluate_roc
from exp_utils.dataset_process_util.img_preprocess import get_embeded_prompt, get_encoded_latents

def get_xt(scheduler, x0, t, eps):
    """
    x_t = sqrt(alpha_t_bar) * x0 + sqrt(1 - alpha_t_bar) * eps
    """
    noise_level = scheduler.alphas_cumprod.to(torch.float32)
    alpha_t_bar = noise_level[t].to(x0.device)
    a_t = alpha_t_bar.sqrt()
    b_t = (1 - alpha_t_bar).sqrt()

    while len(a_t.shape) < len(x0.shape):
        a_t = a_t.unsqueeze(-1)
        b_t = b_t.unsqueeze(-1)

    return a_t * x0 + b_t * eps


def execute_attack_step(pipe,
                        x_0_latents,
                        prompt_embeds,
                        encode_prompt_addtions,
                        attack_t,
                        lp_norm):
    device = x_0_latents.device

    # step 1: get εθ(x0, 0)
    t_0 = torch.tensor([0], device=device, dtype=torch.long)
    eps_0 = pipe.unet(**get_unet_kwargs(
                    pipe=pipe, 
                    encode_prompt_addtions=encode_prompt_addtions,
                    sample=x_0_latents,
                    timestep=t_0,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False))[0]

    # step 2: get εθ(xt, t)
    # timesteps_to_attack = torch.arange(interval, interval * attack_num + 1, interval, device=device)
    timesteps_to_attack = torch.tensor([attack_t], device=device, dtype=torch.long)

    eps_t_list = []
    for t_val in timesteps_to_attack:
        t_input_for_xt = t_val.reshape(1)
        x_t = pipe.scheduler.add_noise(x_0_latents, eps_0, t_input_for_xt)

        t_input_for_unet = t_val.expand(x_t.shape[0])
        eps_t = pipe.unet(**get_unet_kwargs(pipe=pipe, 
                                            encode_prompt_addtions=encode_prompt_addtions,
                                            sample=x_t,
                                            timestep=t_input_for_unet,
                                            encoder_hidden_states=prompt_embeds,
                                            return_dict=False))[0]
        eps_t_list.append(eps_t)

    eps_t_tensor = torch.stack(eps_t_list, dim=0) # shape: [num_timesteps, B, C, H, W]

    # step 3: calculate errors for each time step
    # R_t,p = ||εθ(x0, 0) - εθ(xt, t)||_p (9)
    scores_tensor = ((eps_0.unsqueeze(0) - eps_t_tensor).abs() ** lp_norm).flatten(2).sum(dim=-1) # [num_timesteps, B]

    return -scores_tensor.detach().cpu().numpy() # let `larger_is_member` be True


def execute_piami_attack(dataloader, pipe):
    all_scores = []
    with tqdm(range(len(dataloader)), desc="Execute piami attack") as tqdm_bar:
        for batch in dataloader:
            index, prompt, input_image = batch['index'][0], batch['prompt'], batch['image']
            x_0_latents = get_encoded_latents(input_image.to(PIPE_TYPE), pipe)
            prompt_embeds, encode_prompt_addtions = get_embeded_prompt(pipe, prompt)

            scores = execute_attack_step(pipe,
                                         x_0_latents,
                                         prompt_embeds,
                                         encode_prompt_addtions,
                                         attack_t=200,
                                         lp_norm=4)
            all_scores.append(scores)

            # update progress bar
            tqdm_bar.update(1)

    return np.concatenate(all_scores, axis=1).tolist()[0] # shape: [num_timesteps, total_samples]




def piami_main():
    pipe = txt2img_pipeline_select(attack_args.base_model_id, PIPE_TYPE)

    # pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    train_save_dir = os.path.join(TRAIN_OUTPUT_DIR, "piami")
    hold_out_save_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, "piami")
    res_dir = os.path.join(MI_DIR, "piami")
    member_scores_file = os.path.join(train_save_dir, "piami_member_scores.json")
    nonmember_scores_file = os.path.join(hold_out_save_dir, "piami_nonmember_scores.json")

    if os.path.exists(member_scores_file) and os.path.exists(nonmember_scores_file):
        print(f"attack outputs already exists, skipping inference. Delete the old output directory if meeting any problems when re-running the attack.")
        with open(member_scores_file, "r", encoding='utf-8') as f:
            member_scores = json.load(f)['member_scores']
        with open(nonmember_scores_file, "r", encoding='utf-8') as f:
            nonmember_scores = json.load(f)['nonmember_scores']
    else:
        shutil.rmtree(train_save_dir, ignore_errors=True)
        shutil.rmtree(hold_out_save_dir, ignore_errors=True)
        shutil.rmtree(res_dir, ignore_errors=True)

        train_loader, _ = create_dataset_dataloader(train_save_dir,
                                                    dataset_config_1,
                                                    pipe.tokenizer,
                                                    img_resolution=attack_args.generate_resolution,
                                                    batch_size=1,
                                                    need_save=False)

        hold_out_loader, _ = create_dataset_dataloader(hold_out_save_dir,
                                                       dataset_config_2,
                                                       pipe.tokenizer,
                                                       img_resolution=attack_args.generate_resolution,
                                                       batch_size=1,
                                                       need_save=False)


        member_scores = execute_piami_attack(train_loader, pipe)
        nonmember_scores = execute_piami_attack(hold_out_loader, pipe)

        os.makedirs(train_save_dir, exist_ok=True)
        with open(member_scores_file, "w", encoding='utf-8') as f:
            json.dump({"member_scores": member_scores}, f)
        os.makedirs(hold_out_save_dir, exist_ok=True)
        with open(nonmember_scores_file, "w", encoding='utf-8') as f:
            json.dump({"nonmember_scores": nonmember_scores}, f)

    evaluate_roc(member_scores, nonmember_scores, res_dir, larger_is_member=True)
