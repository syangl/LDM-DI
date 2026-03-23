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


def execute_attack_step(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, timesteps, device):
    """
    Compute denoising loss at multiple different timesteps
    Training samples will have lower denoising loss across multiple timesteps
    
    Returns:
        Negative loss values (larger_is_member=True)
    """
    bs = x_0_latents.shape[0]
    # generate random noise
    noise = torch.randn_like(x_0_latents).to(device)
    losses = []
    
    for timestep in timesteps:
        # add noise to x_0 up to timestep t to get x_t
        t_input_for_xt = timestep.unsqueeze(0)
        x_t = pipe.scheduler.add_noise(x_0_latents, noise, t_input_for_xt)
        
        # predict noise at timestep t
        t_input_for_unet = timestep.expand(x_t.shape[0])
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=x_t,
            timestep=t_input_for_unet,
            encoder_hidden_states=prompt_embeds,
            return_dict=False))[0]
        
        # compute L2 distance between predicted noise and true noise
        loss = torch.norm(
            noise_pred.reshape(bs, -1) - noise.reshape(bs, -1), dim=-1, p=2
        ).cpu()
        
        losses.append(loss)
    
    # stack losses from all timesteps
    losses = torch.stack(losses, dim=1).unsqueeze(2)
    
    # larger_is_member=True
    return -losses.detach().cpu().numpy()


def execute_multiple_lossmi_attack(dataloader, pipe):
    all_scores = []
    # from 0 to 900, every 100 timesteps, 10 timesteps total
    # timesteps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    timesteps = [0, 200, 400, 600, 800, ] # reduce cost
    # average over 5 repetitions
    # n_repetitions = 5
    n_repetitions = 4
    
    with tqdm(range(len(dataloader)), desc="Execute multiple_lossmi attack") as tqdm_bar:
        for batch in dataloader:
            index, prompt, input_image = batch['index'][0], batch['prompt'], batch['image']
            x_0_latents = get_encoded_latents(input_image.to(PIPE_TYPE), pipe)
            prompt_embeds, encode_prompt_addtions = get_embeded_prompt(pipe, prompt)
            device = x_0_latents.device
            timesteps_tensor = torch.tensor(timesteps, device=device, dtype=torch.long)
            
            repetition_scores = []
            # repeat n_repetitions times, each with different random noise
            for _ in range(n_repetitions):
                scores = execute_attack_step(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, timesteps_tensor, device)
                repetition_scores.append(scores)
            
            # average results from multiple repetitions
            repetition_scores = np.array(repetition_scores)
            all_scores.append(repetition_scores.mean(axis=0))
            
            tqdm_bar.update(1)
    
    # # average scores across all timesteps and all samples
    # return np.concatenate(all_scores, axis=0).mean(axis=(1, 2)).tolist()
    
    # concatenate all samples, then reshape to (N_samples, N_timesteps)
    all_scores_np = np.concatenate(all_scores, axis=0)  # (N_samples, N_timesteps, 1)
    all_scores_np = all_scores_np.squeeze(axis=2)  # (N_samples, N_timesteps)
    return all_scores_np.tolist()


def multiple_lossmi_main():
    pipe = txt2img_pipeline_select(attack_args.base_model_id, PIPE_TYPE)
    
    train_save_dir = os.path.join(TRAIN_OUTPUT_DIR, "multiple_lossmi")
    hold_out_save_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, "multiple_lossmi")
    res_dir = os.path.join(MI_DIR, "multiple_lossmi")
    member_scores_file = os.path.join(train_save_dir, "multiple_lossmi_member_scores.json")
    nonmember_scores_file = os.path.join(hold_out_save_dir, "multiple_lossmi_nonmember_scores.json")
    
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
        
        member_scores = execute_multiple_lossmi_attack(train_loader, pipe)
        nonmember_scores = execute_multiple_lossmi_attack(hold_out_loader, pipe)
        
        os.makedirs(train_save_dir, exist_ok=True)
        with open(member_scores_file, "w", encoding='utf-8') as f:
            json.dump({"member_scores": member_scores}, f)
        os.makedirs(hold_out_save_dir, exist_ok=True)
        with open(nonmember_scores_file, "w", encoding='utf-8') as f:
            json.dump({"nonmember_scores": nonmember_scores}, f)
    
    evaluate_roc(member_scores, nonmember_scores, res_dir, larger_is_member=True)
