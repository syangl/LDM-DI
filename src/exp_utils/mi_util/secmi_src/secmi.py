"""
The source code of SecMI https://github.com/jinhaoduan/SecMI-LDM
We only modified the dataset loading code to load our datasets. 
"""

import os
from tqdm import tqdm
import torch
import numpy as np
import random
from exp_utils.config_util.attack_config import get_unet_kwargs, TRAIN_OUTPUT_DIR, dataset_config_1, HOLD_OUT_OUTPUT_DIR, dataset_config_2, MI_DIR, attack_args, PIPE_TYPE
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.metric_util.evaluate import evaluate_roc
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select
from exp_utils.dataset_process_util.img_preprocess import get_embeded_prompt, get_encoded_latents
import json
import shutil




def prepare_extra_step_kwargs(scheduler, generator, eta):
    import inspect
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def single_reverse(
    scheduler,
    model_output,
    timestep,
    prev_timestep,
    sample,
    use_clipped_model_output=False,
):
    """
    Single reverse code. Modified from the implement of https://github.com/jinhaoduan/SecMI-LDM.
    """
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        # predict V
        model_output = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`")

    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    std_dev_t = 0

    if use_clipped_model_output:
        model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * model_output

    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    return prev_sample


def single_step(
    scheduler,
    model_output,
    timestep,
    prev_timestep,
    sample,
    eta,
    use_clipped_model_output=False,
    generator=None,
    variance_noise=None,
):
    """
    Single step code. Modified from the implement of https://github.com/jinhaoduan/SecMI-LDM.
    """
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    # prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        # predict V
        model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`")

    # 4. Clip "predicted x_0"
    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance**(0.5)

    if use_clipped_model_output:
        # the model_output is always re-derived from the clipped x_0 in Glide
        model_output = (sample - alpha_prod_t**(0.5) * pred_original_sample) / beta_prod_t**(0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2)**(0.5) * model_output

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev**(0.5) * pred_original_sample + pred_sample_direction

    if eta > 0:
        # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
        device = model_output.device
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                " `variance_noise` stays `None`.")

        if variance_noise is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                variance_noise = torch.randn(model_output.shape,
                                             dtype=model_output.dtype,
                                             generator=generator)
                variance_noise = variance_noise.to(device)
            else:
                variance_noise = torch.randn(model_output.shape,
                                             generator=generator,
                                             device=device,
                                             dtype=model_output.dtype)
        variance = scheduler._get_variance(
            timestep, prev_timestep)**(0.5) * eta * variance_noise

        prev_sample = prev_sample + variance

    return prev_sample

def reverse_denoise(
    pipe,
    prompt,
    encode_prompt_addtions,
    text_embeddings=None,
    height=None,
    width=None,
    num_inference_steps=20,
    guidance_scale=7.5,
    eta=0.0,
    generator=None,
    latents=None,
):
    """
    Reverse denoise process code. Modified from the implement of https://github.com/jinhaoduan/SecMI-LDM.
    """
    # 0. Default height and width to unet
    # height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    # width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    # self.check_inputs(prompt, height, width, callback_steps)

    # 2. Define call parameters
    # batch_size = 1 if isinstance(prompt, str) else len(prompt)
    batch_size = 1
    device = pipe._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    if prompt is not None:
        timesteps = prompt
    else:
        timesteps = list(range(0, 200, 50))

    # extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    extra_step_kwargs = prepare_extra_step_kwargs(pipe.scheduler, generator, eta)

    original_latents = latents.detach().clone()
    reverse_results = [original_latents]
    for i, t in enumerate(timesteps[:-1]):
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)
        # predict the noise residual
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe,
            encode_prompt_addtions=encode_prompt_addtions, 
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=text_embeddings,
            return_dict=False,))[0]
        # compute the previous noisy sample x_t -> x_t-1
        latents = single_reverse(scheduler=pipe.scheduler,
                                 model_output=noise_pred,
                                 timestep=t,
                                 prev_timestep=timesteps[i + 1],
                                 sample=latents)

        reverse_results.append(latents.detach().clone())  # reverse_results = [x_0, x_50, x_100, x_150]

    timesteps = list(reversed(timesteps))
    denoising_results = []
    for i, t in enumerate(timesteps[:-1]):

        latents = reverse_results[len(reverse_results) - 1 - i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=text_embeddings,
            return_dict=False,))[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = single_step(scheduler=pipe.scheduler,
                              model_output=noise_pred,
                              timestep=t,
                              prev_timestep=timesteps[i + 1],
                              sample=latents,
                              **extra_step_kwargs)

        denoising_results.append(latents.detach().clone()) # denoising_results = [x_150, x_100, x_50, x_0]

    return reverse_results, denoising_results




def get_reverse_denoise_results(pipe, dataloader, prefix='member'):
    mean_l2 = 0
    scores = []
    with tqdm(range(len(dataloader)), desc="Get reverse denoise results") as tqdm_bar:
        for batch_idx, batch in enumerate(dataloader):
            # Convert images to latent space
            pixel_values = batch["image"].to(PIPE_TYPE)
            pixel_values = pixel_values.cuda()
            # latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            # latents = latents * 0.18215
            latents = get_encoded_latents(pixel_values, pipe)
            # Get the text embedding for conditioning
            # input_ids = batch["input_ids"].cuda()
            # encoder_hidden_states = pipe.text_encoder(input_ids)[0]
            prompt_embeds, encode_prompt_addtions = get_embeded_prompt(pipe, batch["prompt"])

            reverse_results, denoising_results = reverse_denoise(
                pipe=pipe,
                prompt=None,
                latents=latents,
                text_embeddings=prompt_embeds,
                encode_prompt_addtions=encode_prompt_addtions,
                guidance_scale=1.0)
            # pipe(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)

            # score = ((denoising_results[-15] - reverse_results[14])**2).sum() # use x_140
            score = ((denoising_results[-3] - reverse_results[2])**2).sum()
            scores.append(score.reshape(-1, 1))
            mean_l2 += score
            tqdm_bar.set_description(f'mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')
            tqdm_bar.update(1)

    return torch.concat(scores).reshape(-1)


def secmi_main():
    # pipe = load_pipeline(attack_args.base_model_id, "cuda")
    pipe = txt2img_pipeline_select(base_model_id=attack_args.base_model_id, PIPE_TYPE=PIPE_TYPE)

    # pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    train_save_dir = os.path.join(TRAIN_OUTPUT_DIR, "secmi")
    hold_out_save_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, "secmi")
    res_dir = os.path.join(MI_DIR, "secmi")
    member_scores_file = os.path.join(train_save_dir, "secmi_member_scores.json")
    nonmember_scores_file = os.path.join(hold_out_save_dir, "secmi_nonmember_scores.json")
    if os.path.exists(member_scores_file) and os.path.exists(nonmember_scores_file):
        print(f"attack outputs already exists, skipping inference. Delete the old output directory if meeting any problems when re-running the attack.")
        with open(member_scores_file, "r", encoding='utf-8') as f:
            member_scores = json.load(f)['member_scores']
        with open(nonmember_scores_file, "r", encoding='utf-8') as f:
            nonmember_scores = json.load(f)['nonmember_scores']

    else:
        # delete old outputs
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

        test_loader, _ = create_dataset_dataloader(hold_out_save_dir,
                                                   dataset_config_2,
                                                   pipe.tokenizer,
                                                   img_resolution=attack_args.generate_resolution,
                                                   batch_size=1,
                                                   need_save=False)

        # execute attack
        member_scores = get_reverse_denoise_results(pipe, train_loader)
        nonmember_scores = get_reverse_denoise_results(pipe, test_loader)

        # save results
        member_scores = member_scores.cpu().numpy().tolist()
        nonmember_scores = nonmember_scores.cpu().numpy().tolist()

        os.makedirs(train_save_dir, exist_ok=True)
        with open(member_scores_file, "w", encoding='utf-8') as f:
            json.dump({"member_scores": member_scores}, f)
        os.makedirs(hold_out_save_dir, exist_ok=True)
        with open(nonmember_scores_file, "w", encoding='utf-8') as f:
            json.dump({"nonmember_scores": nonmember_scores}, f)

    evaluate_roc(member_scores,
                 nonmember_scores,
                 os.path.join(MI_DIR, f"secmi"),
                 larger_is_member=False)


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
