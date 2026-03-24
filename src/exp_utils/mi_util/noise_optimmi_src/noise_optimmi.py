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


def execute_attack_step(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, timestep, max_iter, device):
    """
    Use L-BFGS-B optimizer to find the perturbation that minimizes prediction loss
    For training samples, the model can find optimal perturbation faster, and the final perturbation has smaller L2 norm

    Returns:
        Negative loss and perturbation norm (larger_is_member=True)
    """
    bs = x_0_latents.shape[0]
    noise = torch.randn_like(x_0_latents).to(device)
    
    # add noise to x_0 to get x_t
    t_input_for_xt = timestep.unsqueeze(0)
    latents_noised = pipe.scheduler.add_noise(x_0_latents, noise, t_input_for_xt)
        
    # use a simpler approach for optimization to avoid device issues
    perturbation = torch.zeros_like(latents_noised, requires_grad=True)
    optimizer = torch.optim.LBFGS([perturbation], max_iter=max_iter, lr=0.1)
    
    def closure():
        optimizer.zero_grad()
        input_latents = latents_noised + perturbation
        t_input_for_unet = timestep.expand(input_latents.shape[0])
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=input_latents,
            timestep=t_input_for_unet,
            encoder_hidden_states=prompt_embeds,
            return_dict=False))[0]
        
        loss = torch.norm(
            noise_pred.reshape(bs, -1) - noise.reshape(bs, -1), dim=-1, p=2
        ).mean()
        
        loss.backward()
        return loss
    
    # perform optimization
    optimizer.step(closure)
    
    # get final results
    with torch.no_grad():
        final_input = latents_noised + perturbation
        t_input_for_unet = timestep.expand(final_input.shape[0])
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=final_input,
            timestep=t_input_for_unet,
            encoder_hidden_states=prompt_embeds,
            return_dict=False))[0]
        
        loss = torch.norm(
            noise_pred.reshape(bs, -1) - noise.reshape(bs, -1), dim=-1, p=2
        ).cpu()
        
        l2_perturbations = torch.norm(perturbation.view(bs, -1), dim=-1, p=2).cpu()
        
        # replace NaN values before returning
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
        l2_perturbations = torch.nan_to_num(l2_perturbations, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # return negative values for larger_is_member=True
    return -torch.stack([loss, l2_perturbations], dim=1).unsqueeze(1).detach().cpu().numpy()


def execute_noise_optimmi_attack(dataloader, pipe):
    all_scores_loss = []
    all_scores_l2 = []
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
            
            # scores shape: [B, 1, 2] where B=1, so:
            # scores[0, 0, 0] = loss
            # scores[0, 0, 1] = l2_perturbation
            all_scores_loss.append(scores[0, 0, 0])
            all_scores_l2.append(scores[0, 0, 1])
            
            tqdm_bar.update(1)
    
    # return a 2d list where each row has two features [loss, l2]
    result = []
    for loss, l2 in zip(all_scores_loss, all_scores_l2):
        loss_val = loss.item()
        l2_val = l2.item()
        
        # Replace NaN with reasonable defaults
        if np.isnan(loss_val):
            loss_val = 0.0
        if np.isnan(l2_val):
            l2_val = 0.0
            
        result.append([loss_val, l2_val])
    
    return result


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
