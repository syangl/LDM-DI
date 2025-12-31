"""
This script is used to evaluate the baseline: Structural MIA.
The attack code is from the official repo: https://github.com/MM2024submission2850/StructualMIA
We modified the model loader and dataset loader to use Huggingface diffusers pipeline module and our experimental datasets.
"""



import torch
import sys, os
# get the current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
# add the project root directory as the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_utils.config_util.attack_config import attack_args, get_unet_kwargs, PIPE_TYPE, TRAIN_OUTPUT_DIR, HOLD_OUT_OUTPUT_DIR, dataset_config_1, dataset_config_2, MI_DIR
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.metric_util.evaluate import evaluate_roc, calculate_similarity
import json
import shutil
from exp_utils.dataset_process_util.img_preprocess import latent_decode, pil2tensor, get_encoded_latents, get_embeded_prompt

def ddim_singlestep_pipe(model_output, x_c, t_c, t_target, pipe):
    """
    A huggingface DDIMSampler like singel step function.
    """
    alpha_prod_t_c = pipe.scheduler.alphas_cumprod[t_c]
    alpha_prod_t_target = pipe.scheduler.alphas_cumprod[t_target] if t_target >= 0 else  pipe.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t_c

    if pipe.scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (x_c - beta_prod_t ** (0.5) * model_output) / alpha_prod_t_c ** (0.5)
    elif pipe.scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif pipe.scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t_c ** 0.5) * x_c - (beta_prod_t ** 0.5) * model_output
        # predict V
        model_output = (alpha_prod_t_c ** 0.5) * model_output + (beta_prod_t ** 0.5) * x_c
    else:
        raise ValueError(
            f"prediction_type given as { pipe.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`")

    if pipe.scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    std_dev_t = 0

    target_sample_direction = (1 - alpha_prod_t_target - std_dev_t ** 2) ** (0.5) * model_output

    target_sample = alpha_prod_t_target ** (0.5) * pred_original_sample + target_sample_direction

    return target_sample

def naive_inversion_pipe(pipe, x_0, prompt_embeds, encode_prompt_addtions):
    x = x_0
    t_c = 0
    target_steps = list(range(0, 150, 50))[1:]

    for t_targets in target_steps:
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=x,
            timestep=t_c,
            encoder_hidden_states=prompt_embeds,
            return_dict=False))[0]
        result = ddim_singlestep_pipe(noise_pred, x, t_c, t_targets, pipe)
        x = result
        t_c = t_targets
    x_sec = x

    return x_sec



def execute_structuralmi_attack(dataloader, pipe, choice="ssim", **kwargs):
    ssim_scores = []
    with torch.no_grad():
        dataset_images = []
        output_images = []
        for batch in dataloader:
            index, prompt, input_image = batch['index'][0], batch['prompt'], batch['image']
            dataset_images.append(input_image.squeeze(0))
            clean_latents = get_encoded_latents(input_image.to(PIPE_TYPE), pipe)
            # embedding prompt
            prompt_embeds, encode_prompt_addtions = get_embeded_prompt(pipe, prompt)
            # ddim inversion
            x_T = naive_inversion_pipe(pipe=pipe,
                                       x_0=clean_latents,
                                       prompt_embeds=prompt_embeds,
                                       encode_prompt_addtions=encode_prompt_addtions,
                                       )
            generator = torch.Generator("cuda").manual_seed(attack_args.global_seed)
            # decode
            image_T = latent_decode(x_T, pipe, generator)
            output_images.append([pil2tensor(image_T),])

            # save image_T for quick eval if needed
            save_dir = kwargs.get('save_dir', None)
            if save_dir is not None:
                save_dir = os.path.join(save_dir, index)
                os.makedirs(save_dir, exist_ok=True)
                image_T.save(os.path.join(save_dir, f"{index}.jpg"))

            # calculate ssim
            # ssim = cal_loss(input_image.squeeze(0), image_T).item()
            # ssim_scores.append(ssim)
        ssim_scores = calculate_similarity(output_images, dataset_images, choice=choice)
    return ssim_scores

def structuralmi_main():
    pipe = txt2img_pipeline_select(attack_args.base_model_id, PIPE_TYPE)

    # pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    train_save_dir = os.path.join(TRAIN_OUTPUT_DIR, "structuralmi")
    hold_out_save_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, "structuralmi")
    res_dir = os.path.join(MI_DIR, "structuralmi")
    member_scores_file = os.path.join(train_save_dir, "structuralmi_member_scores.json")
    nonmember_scores_file = os.path.join(hold_out_save_dir, "structuralmi_nonmember_scores.json")
    if os.path.exists(member_scores_file) and os.path.exists(nonmember_scores_file):
        print(f"attack outputs already exists, skipping inference. Delete the old output directory if meeting any problems when re-running the attack.")
        with open(member_scores_file, "r", encoding='utf-8') as f:
            member_scores = json.load(f)['member_scores']
        with open(nonmember_scores_file, "r", encoding='utf-8') as f:
            nonmember_scores = json.load(f)['nonmember_scores']
    else:
        # delete old output directory
        shutil.rmtree(train_save_dir, ignore_errors=True)
        shutil.rmtree(hold_out_save_dir, ignore_errors=True)
        shutil.rmtree(res_dir, ignore_errors=True)

        # create dataset
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

        # execute attack
        member_scores = execute_structuralmi_attack(train_loader, pipe)
        nonmember_scores = execute_structuralmi_attack(hold_out_loader, pipe)

        # save results
        os.makedirs(train_save_dir, exist_ok=True)
        with open(member_scores_file, "w", encoding='utf-8') as f:
            json.dump({"member_scores": member_scores}, f)
        os.makedirs(hold_out_save_dir, exist_ok=True)
        with open(nonmember_scores_file, "w", encoding='utf-8') as f:
            json.dump({"nonmember_scores": nonmember_scores}, f)

    evaluate_roc(member_scores, nonmember_scores, res_dir, larger_is_member=True)
