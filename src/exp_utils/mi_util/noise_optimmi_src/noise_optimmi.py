import os, sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import json
import shutil
from tqdm import tqdm
from scipy.optimize import minimize
from exp_utils.config_util.attack_config import attack_args, get_unet_kwargs, PIPE_TYPE, TRAIN_OUTPUT_DIR, HOLD_OUT_OUTPUT_DIR, dataset_config_1, dataset_config_2, MI_DIR
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.metric_util.evaluate import evaluate_roc
from exp_utils.dataset_process_util.img_preprocess import get_embeded_prompt, get_encoded_latents


def obj(perturbations, B, latents_noised, prompt_embeds, encode_prompt_addtions, timestep, noise, pipe, device):
    perturbations = torch.from_numpy(perturbations).requires_grad_(True)
    input_latents = latents_noised + perturbations.reshape(*latents_noised.shape)
    input_latents = input_latents.to(device).float()
    noise = noise.to(device).float()
    
    t_input_for_unet = timestep.expand(input_latents.shape[0])
    noise_pred = pipe.unet(**get_unet_kwargs(
        pipe=pipe, 
        encode_prompt_addtions=encode_prompt_addtions,
        sample=input_latents,
        timestep=t_input_for_unet,
        encoder_hidden_states=prompt_embeds,
        return_dict=False))[0]
    
    loss = torch.norm(
        noise_pred.reshape(B, -1) - noise.reshape(B, -1), dim=-1, p=2
    ).mean()
    loss.backward()
    gradient = perturbations.grad.view(-1)
    
    return loss.item(), gradient.cpu().numpy()


def execute_attack_step(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, timestep, max_iter, device):
    bs = x_0_latents.shape[0]
    noise = torch.randn_like(x_0_latents).to(device)
    t_input_for_xt = timestep.unsqueeze(0)
    latents_noised = pipe.scheduler.add_noise(x_0_latents, noise, t_input_for_xt)
    
    out = minimize(
        obj,
        torch.zeros_like(latents_noised).cpu().reshape(-1).numpy(),
        args=(
            bs,
            latents_noised.cpu(),
            prompt_embeds.cpu(),
            encode_prompt_addtions,
            timestep.cpu(),
            noise.cpu(),
            pipe,
            device,
        ),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter},
    )
    
    perturbations = torch.from_numpy(out.x.reshape(*latents_noised.shape)).to(device)
    
    t_input_for_unet = timestep.expand((latents_noised + perturbations).shape[0])
    noise_pred = pipe.unet(**get_unet_kwargs(
        pipe=pipe, 
        encode_prompt_addtions=encode_prompt_addtions,
        sample=(latents_noised + perturbations).float(),
        timestep=t_input_for_unet,
        encoder_hidden_states=prompt_embeds,
        return_dict=False))[0]
    
    loss = torch.norm(
        noise_pred.reshape(bs, -1) - noise.reshape(bs, -1), dim=-1, p=2
    ).cpu()
    
    l2_perturbations = torch.norm(perturbations.view(bs, -1), dim=-1, p=2).cpu()
    
    return -torch.stack([loss, l2_perturbations], dim=1).unsqueeze(1).detach().cpu().numpy()


def execute_noise_optimmi_attack(dataloader, pipe):
    all_scores = []
    timestep = 100
    max_iter = 5
    with tqdm(range(len(dataloader)), desc="Execute noise_optimmi attack") as tqdm_bar:
        for batch in dataloader:
            index, prompt, input_image = batch['index'][0], batch['prompt'], batch['image']
            x_0_latents = get_encoded_latents(input_image.to(PIPE_TYPE), pipe)
            prompt_embeds, encode_prompt_addtions = get_embeded_prompt(pipe, prompt)
            device = x_0_latents.device
            timestep_tensor = torch.tensor([timestep], device=device, dtype=torch.long)
            scores = execute_attack_step(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, timestep_tensor[0], max_iter, device)
            all_scores.append(scores)
            tqdm_bar.update(1)
    return np.concatenate(all_scores, axis=0).mean(axis=(1, 2)).tolist()


def noise_optimmi_main():
    pipe = txt2img_pipeline_select(attack_args.base_model_id, PIPE_TYPE)
    
    train_save_dir = os.path.join(TRAIN_OUTPUT_DIR, "noise_optimmi")
    hold_out_save_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, "noise_optimmi")
    res_dir = os.path.join(MI_DIR, "noise_optimmi")
    member_scores_file = os.path.join(train_save_dir, "noise_optimmi_member_scores.json")
    nonmember_scores_file = os.path.join(hold_out_save_dir, "noise_optimmi_nonmember_scores.json")
    
    if os.path.exists(member_scores_file) and os.path.exists(nonmember_scores_file):
        print(f"attack outputs already exists, skipping inference.")
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
        
        member_scores = execute_noise_optimmi_attack(train_loader, pipe)
        nonmember_scores = execute_noise_optimmi_attack(hold_out_loader, pipe)
        
        os.makedirs(train_save_dir, exist_ok=True)
        with open(member_scores_file, "w", encoding='utf-8') as f:
            json.dump({"member_scores": member_scores}, f)
        os.makedirs(hold_out_save_dir, exist_ok=True)
        with open(nonmember_scores_file, "w", encoding='utf-8') as f:
            json.dump({"nonmember_scores": nonmember_scores}, f)
    
    evaluate_roc(member_scores, nonmember_scores, res_dir, larger_is_member=True)
