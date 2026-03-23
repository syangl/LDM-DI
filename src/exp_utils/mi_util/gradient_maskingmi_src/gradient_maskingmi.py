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
from exp_utils.config_util.attack_config import attack_args, get_unet_kwargs, PIPE_TYPE, TRAIN_OUTPUT_DIR, HOLD_OUT_OUTPUT_DIR, dataset_config_1, dataset_config_2, MI_DIR
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.metric_util.evaluate import evaluate_roc
from exp_utils.dataset_process_util.img_preprocess import get_embeded_prompt, get_encoded_latents


def obtain_gradients(pipe, latents, prompt_embeds, encode_prompt_addtions, noise, timesteps, device):
    out = []
    for timestep in timesteps:
        inputs = pipe.scheduler.add_noise(latents, noise, timestep.unsqueeze(0)).clone().detach().requires_grad_(True)
        t_input_for_unet = timestep.expand(inputs.shape[0])
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=inputs,
            timestep=t_input_for_unet,
            encoder_hidden_states=prompt_embeds,
            return_dict=False))[0]
        loss = torch.norm(noise_pred - noise, dim=(1, 2, 3), p=2).mean()
        loss.backward()
        out.append(inputs.grad)
    return torch.stack(out, dim=1)


def get_masks(gradients, B, masking_ratio):
    out = []
    k = int(len(gradients[0][0].flatten()) * masking_ratio)
    for gradient in gradients.permute(1, 0, 2, 3, 4):
        _, i = torch.topk(gradient.view(B, -1).abs(), k=k, dim=1)
        indices = torch.stack(torch.unravel_index(i, gradient[0].shape)).cpu()
        indices = torch.concatenate(
            [
                torch.arange(B).repeat_interleave(k).unsqueeze(0),
                indices.view(len(gradient.shape) - 1, B * k),
            ],
            dim=0,
        )
        mask = torch.zeros_like(gradient)
        mask[list(indices)] = 1
        out.append(mask.bool())
    return torch.stack(out, dim=1)


def noise_top_values(latents, masks, noise):
    out = []
    for mask in masks.permute(1, 0, 2, 3, 4):
        inputs = latents.clone().detach()
        inputs[mask] = noise[mask]
        out.append(inputs)
    return torch.stack(out, dim=1)


def masked_noise_pred(pipe, latents_noised, prompt_embeds, encode_prompt_addtions, timesteps):
    out = []
    for idx, timestep in enumerate(timesteps):
        t_input_for_unet = timestep.expand(latents_noised[:, idx].shape[0])
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=latents_noised[:, idx],
            timestep=t_input_for_unet,
            encoder_hidden_states=prompt_embeds,
            return_dict=False))[0]
        out.append(noise_pred)
    return torch.stack(out, dim=1)


def get_final_loss(masks, latents, noise, noise_preds, B):
    out = []
    for mask, noise_pred in zip(
        masks.permute(1, 0, 2, 3, 4), noise_preds.permute(1, 0, 2, 3, 4)
    ):
        diff = ((noise - latents)[mask] - noise_pred[mask]).reshape(B, -1)
        out.append(torch.norm(diff, dim=1, p=2).unsqueeze(1))
    return torch.stack(out, dim=1)


def execute_attack_step(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, timesteps, masking_ratio):
    device = x_0_latents.device
    B = x_0_latents.shape[0]
    noise = torch.randn_like(x_0_latents).to(device)
    timesteps_tensor = torch.tensor(timesteps, device=device, dtype=torch.long)
    
    gradients = obtain_gradients(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, noise, timesteps_tensor, device)
    masks = get_masks(gradients, B, masking_ratio)
    latents_noised = noise_top_values(x_0_latents, masks, noise)
    noise_preds = masked_noise_pred(pipe, latents_noised, prompt_embeds, encode_prompt_addtions, timesteps_tensor)
    final_loss = get_final_loss(masks, x_0_latents, noise, noise_preds, B)
    
    return -final_loss.detach().cpu().numpy()


def execute_gradient_maskingmi_attack(dataloader, pipe):
    all_scores = []
    timesteps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    masking_ratio = 0.2
    with tqdm(range(len(dataloader)), desc="Execute gradient_maskingmi attack") as tqdm_bar:
        for batch in dataloader:
            index, prompt, input_image = batch['index'][0], batch['prompt'], batch['image']
            x_0_latents = get_encoded_latents(input_image.to(PIPE_TYPE), pipe)
            prompt_embeds, encode_prompt_addtions = get_embeded_prompt(pipe, prompt)
            scores = execute_attack_step(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, timesteps, masking_ratio)
            all_scores.append(scores)
            tqdm_bar.update(1)
    return np.concatenate(all_scores, axis=0).mean(axis=1).tolist()


def gradient_maskingmi_main():
    pipe = txt2img_pipeline_select(attack_args.base_model_id, PIPE_TYPE)
    
    train_save_dir = os.path.join(TRAIN_OUTPUT_DIR, "gradient_maskingmi")
    hold_out_save_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, "gradient_maskingmi")
    res_dir = os.path.join(MI_DIR, "gradient_maskingmi")
    member_scores_file = os.path.join(train_save_dir, "gradient_maskingmi_member_scores.json")
    nonmember_scores_file = os.path.join(hold_out_save_dir, "gradient_maskingmi_nonmember_scores.json")
    
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
        
        member_scores = execute_gradient_maskingmi_attack(train_loader, pipe)
        nonmember_scores = execute_gradient_maskingmi_attack(hold_out_loader, pipe)
        
        os.makedirs(train_save_dir, exist_ok=True)
        with open(member_scores_file, "w", encoding='utf-8') as f:
            json.dump({"member_scores": member_scores}, f)
        os.makedirs(hold_out_save_dir, exist_ok=True)
        with open(nonmember_scores_file, "w", encoding='utf-8') as f:
            json.dump({"nonmember_scores": nonmember_scores}, f)
    
    evaluate_roc(member_scores, nonmember_scores, res_dir, larger_is_member=True)
