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
import json
import shutil
from tqdm import tqdm
from exp_utils.config_util.attack_config import attack_args, get_unet_kwargs, PIPE_TYPE, TRAIN_OUTPUT_DIR, HOLD_OUT_OUTPUT_DIR, dataset_config_1, dataset_config_2, MI_DIR
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.metric_util.evaluate import evaluate_roc
from exp_utils.dataset_process_util.img_preprocess import get_embeded_prompt, get_encoded_latents, pil2tensor, latent_decode
from exp_utils.metric_util.evaluate import calculate_similarity




def execute_rediffusemi_attack(dataloader, pipe, choice="ssim"):
    def ddim_single_step(model_output, x_c, t_c, t_target, pipe):
        """
        A huggingface DDIMSampler like singel step function.
        """
        alpha_prod_t_c = pipe.scheduler.alphas_cumprod[t_c]
        alpha_prod_t_target = pipe.scheduler.alphas_cumprod[t_target] if t_target >= 0 else pipe.scheduler.final_alpha_cumprod

        beta_prod_t_c = 1 - alpha_prod_t_c

        if pipe.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (x_c - beta_prod_t_c**(0.5) * model_output) / alpha_prod_t_c**(0.5)
        elif pipe.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif pipe.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t_c**0.5) * x_c - (beta_prod_t_c**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t_c**0.5) * model_output + (beta_prod_t_c**0.5) * x_c
        else:
            raise ValueError(
                f"prediction_type given as { pipe.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`")

        if pipe.scheduler.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        std_dev_t = 0

        target_sample_direction = (1 - alpha_prod_t_target - std_dev_t**2)**(0.5) * model_output

        target_sample = alpha_prod_t_target**(0.5) * pred_original_sample + target_sample_direction

        return target_sample

    def ddim_forward_step(pipe,
                          x_t,
                          custom_timesteps,
                          prompt_embeds,
                          encode_prompt_addtions,
                          std_noise):
        # current timestep
        x_c = x_t
        t_c = custom_timesteps[0]
        with torch.no_grad():
            for i, t_target in enumerate(custom_timesteps[1:]):
                # get model output noise
                noise_pred_c = pipe.unet(**get_unet_kwargs(
                    pipe=pipe, 
                    encode_prompt_addtions=encode_prompt_addtions,
                    sample=x_c,
                    timestep=t_c,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                ))[0]
                # ddim singe step
                result_c = ddim_single_step(noise_pred_c, x_c, t_c, t_target, pipe)
                x_c = result_c
                t_c = t_target

            # get target x_s
            x_s = x_c
        return x_s

    def get_init_x_t(pipe, x_0_latents, t_timestep, generator):
        # for the standard inference, we use a random guassian noise
        noise = torch.randn(x_0_latents.shape,
                            generator=generator,
                            dtype=x_0_latents.dtype,
                            device=x_0_latents.device)
        x_t_latents = pipe.scheduler.add_noise(x_0_latents, noise, t_timestep)
        return x_t_latents, noise
    ################################################################################################################

    # get custom timesteps
    custom_timesteps = list(reversed(range(0, 100 + 1, 50)))

    # Generate images
    dataset_images = []
    output_images = []
    for batch in tqdm(dataloader, desc="generating images...", total=len(dataloader)):
        index, prompt, input_image = batch['index'][0], batch['prompt'], batch['image']

        # save original image
        dataset_images.append(input_image.squeeze(0))

        # get clean latents
        x_0_latents = get_encoded_latents(input_image.to(PIPE_TYPE), pipe)

        # embedding prompt
        prompt_embeds, encode_prompt_addtions = get_embeded_prompt(pipe, prompt)

        x_all_latents = []
        for i in range(5): # let n = 5
            generator = torch.Generator("cuda").manual_seed(attack_args.global_seed + i)
            # get x_t latents
            x_t_latents, std_noise = get_init_x_t(pipe, x_0_latents, torch.tensor(custom_timesteps[0]), generator)
            # forward
            x_s_latents = ddim_forward_step(
                pipe=pipe,
                x_t=x_t_latents,
                custom_timesteps=custom_timesteps,
                prompt_embeds=prompt_embeds,
                encode_prompt_addtions=encode_prompt_addtions,
                std_noise=std_noise,
            )

            x_all_latents.append(x_s_latents)

        # decode average x_s_latents
        x_s_latents_mean = torch.mean(torch.stack(x_all_latents), dim=0)
        image_output = latent_decode(x_s_latents_mean, pipe, generator)
        output_images.append([pil2tensor(image_output),])
    # calculate similarity
    metric_scores = calculate_similarity(output_images, dataset_images, choice=choice)
    return metric_scores




def rediffusemi_main():
    pipe = txt2img_pipeline_select(attack_args.base_model_id, PIPE_TYPE)

    # pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    train_save_dir = os.path.join(TRAIN_OUTPUT_DIR, "rediffusemi")
    hold_out_save_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, "rediffusemi")
    res_dir = os.path.join(MI_DIR, "rediffusemi")
    member_scores_file = os.path.join(train_save_dir, "rediffusemi_member_scores.json")
    nonmember_scores_file = os.path.join(hold_out_save_dir, "rediffusemi_nonmember_scores.json")

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


        member_scores = execute_rediffusemi_attack(train_loader, pipe)
        nonmember_scores = execute_rediffusemi_attack(hold_out_loader, pipe)

        os.makedirs(train_save_dir, exist_ok=True)
        with open(member_scores_file, "w", encoding='utf-8') as f:
            json.dump({"member_scores": member_scores}, f)
        os.makedirs(hold_out_save_dir, exist_ok=True)
        with open(nonmember_scores_file, "w", encoding='utf-8') as f:
            json.dump({"nonmember_scores": nonmember_scores}, f)

    evaluate_roc(member_scores, nonmember_scores, res_dir, larger_is_member=True)
