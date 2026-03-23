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
    """
    Step 1: Compute gradients of the prediction loss with respect to the input latents at each timestep
    Training samples get larger gradients
    
    Args:
        pipe: Diffusion model pipeline
        latents: Original image latent encoding (x_0)
        prompt_embeds: Prompt embeddings
        encode_prompt_addtions: Additional prompt encoding info
        noise: Random noise
        timesteps: List of timesteps to compute
        device: Device
    
    Returns:
        gradients: Gradients at each timestep [B, num_timesteps, C, H, W]
    """
    out = []
    for timestep in timesteps:
        # add noise to x_0 to get x_t
        inputs = pipe.scheduler.add_noise(latents, noise, timestep.unsqueeze(0)).clone().detach().requires_grad_(True)
        
        # predict noise using U-Net
        t_input_for_unet = timestep.expand(inputs.shape[0])
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=inputs,
            timestep=t_input_for_unet,
            encoder_hidden_states=prompt_embeds,
            return_dict=False))[0]
        
        # compute L2 loss between predicted noise and true noise
        loss = torch.norm(noise_pred - noise, dim=(1, 2, 3), p=2).mean()
        
        # backpropagate to compute gradients
        loss.backward()
        
        # save gradients of x_t w.r.t. loss
        out.append(inputs.grad)
    
    return torch.stack(out, dim=1)


def get_masks(gradients, B, masking_ratio):
    """
    Step 2: Create masks based on gradient magnitudes, marking top-k positions with largest gradients
    (Positions with large gradients indicate the model is more sensitive to those positions)
 
    Returns:
        masks: Boolean mask tensor [B, num_timesteps, C, H, W] 
    """
    out = []
    # calculate number of elements to mask: total elements × masking ratio
    k = int(len(gradients[0][0].flatten()) * masking_ratio)
    
    # swap dimensions to iterate by timestep
    for gradient in gradients.permute(1, 0, 2, 3, 4):
        # flatten gradients, take absolute value, find indices of top-k largest gradients
        _, i = torch.topk(gradient.view(B, -1).abs(), k=k, dim=1)
        
        # convert flattened indices back to multi-dimensional indices (C, H, W)
        indices = torch.stack(torch.unravel_index(i, gradient[0].shape)).cpu()
        
        # add batch indices to build complete indices [batch_idx, C, H, W]
        indices = torch.concatenate(
            [
                torch.arange(B).repeat_interleave(k).unsqueeze(0),
                indices.view(len(gradient.shape) - 1, B * k),
            ],
            dim=0,
        )
        
        # create full zero mask, set selected positions to 1
        mask = torch.zeros_like(gradient)
        mask[list(indices)] = 1
        
        # convert to boolean type and save
        out.append(mask.bool())
    
    return torch.stack(out, dim=1)


def noise_top_values(latents, masks, noise):
    """
    Step 3: Replace original latents with random noise at masked positions
    
    
    Returns:
        latents_noised: Masked latents [B, num_timesteps, C, H, W]
    """
    out = []
    for mask in masks.permute(1, 0, 2, 3, 4):
        inputs = latents.clone().detach()
        # replace with noise at masked positions
        inputs[mask] = noise[mask]
        out.append(inputs)
    
    return torch.stack(out, dim=1)


def masked_noise_pred(pipe, latents_noised, prompt_embeds, encode_prompt_addtions, timesteps):
    """
    Step 4: Predict noise using masked latents
    
    Returns:
        noise_preds: Noise predictions at each timestep [B, num_timesteps, C, H, W]
    """
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
    """
    Step 5: Compute final loss, only comparing predicted noise at masked positions
    prediction loss for training samples will be larger than non-training samples

    Args:
        masks: Masks
        latents: Original latents
        noise: True noise
        noise_preds: Predicted noise
        B: Batch size
    
    Returns:
        final_loss: Final loss [B, num_timesteps, 1]
    """
    out = []
    for mask, noise_pred in zip(
        masks.permute(1, 0, 2, 3, 4), noise_preds.permute(1, 0, 2, 3, 4)
    ):
        # compute difference between (noise - latents) and predicted noise at masked positions
        diff = ((noise - latents)[mask] - noise_pred[mask]).reshape(B, -1)
        out.append(torch.norm(diff, dim=1, p=2).unsqueeze(1))
    
    return torch.stack(out, dim=1)


def execute_attack_step(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, timesteps, masking_ratio):
    """
    Execute complete gradient masking attack pipeline
    
    Returns:
        Negative loss values (larger_is_member=True)
    """
    device = x_0_latents.device
    B = x_0_latents.shape[0]
    noise = torch.randn_like(x_0_latents).to(device)
    timesteps_tensor = torch.tensor(timesteps, device=device, dtype=torch.long)
    
    # execute five steps
    gradients = obtain_gradients(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, noise, timesteps_tensor, device)
    masks = get_masks(gradients, B, masking_ratio)
    latents_noised = noise_top_values(x_0_latents, masks, noise)
    noise_preds = masked_noise_pred(pipe, latents_noised, prompt_embeds, encode_prompt_addtions, timesteps_tensor)
    final_loss = get_final_loss(masks, x_0_latents, noise, noise_preds, B)
    
    # return negative loss because we set larger_is_member=True in evaluate_roc
    return -final_loss.detach().cpu().numpy()


def execute_gradient_maskingmi_attack(dataloader, pipe):
    """
    Iterate through entire dataloader and execute gradient masking attack
    """
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
    
    # average scores across all timesteps
    return np.concatenate(all_scores, axis=0).mean(axis=1).tolist()


def gradient_maskingmi_main():
    """
    Gradient Masking main function
    """
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
