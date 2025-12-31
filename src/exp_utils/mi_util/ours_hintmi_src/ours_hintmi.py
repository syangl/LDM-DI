import os, sys
# get the current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# get the project root directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(current_script_dir)))
# add the project root directory as the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_utils.config_util.attack_config import attack_args, get_unet_kwargs, PIPE_TYPE
from exp_utils.metric_util.evaluate import calculate_similarity
import torch
import torch.nn.functional as F
from tqdm import tqdm
from exp_utils.dataset_process_util.img_preprocess import get_embeded_prompt, get_encoded_latents, pil2tensor, latent_decode


#################################################### HintMI ##################################################


def execute_hintmi(dataloader, pipe, choice="ssim"):
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

    def get_init_x_t(pipe, x_0_latents, prompt_embeds, encode_prompt_addtions, t_timestep):
        generator = torch.Generator("cuda").manual_seed(attack_args.global_seed)
        # for the standard inference, we use a random guassian noise
        noise = torch.randn(x_0_latents.shape,
                            generator=generator,
                            dtype=x_0_latents.dtype,
                            device=x_0_latents.device)
        tmp_x_t = pipe.scheduler.add_noise(x_0_latents, noise, t_timestep)
        noise_pred_t = pipe.unet(**get_unet_kwargs(
            pipe=pipe, 
            encode_prompt_addtions=encode_prompt_addtions,
            sample=tmp_x_t,
            timestep=t_timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        ))[0]
        x_t_latents = ddim_single_step(noise_pred_t, x_0_latents, 0, t_timestep, pipe)
        return x_t_latents, noise


    def get_hint_noise(noise_pred, std_noise, timestep_idx, total_timesteps):
        """
            Get the hint noise for current latents at current timestep.
        """
        def get_mix_weight(step_index, total_steps, gamma):
            w = 1 - (step_index / total_steps)**gamma
            return w

        def get_mixed_noise(model_noise_pred, std_noise, hint_weight, strategy="penalty"):
            base_hint_noise = (1 - hint_weight) * model_noise_pred + hint_weight * std_noise
            if "linear" in strategy:
                return base_hint_noise
            elif "penalty" in strategy:
                penalty_lambda = 0.5
                p = F.softmax(model_noise_pred, dim=1)
                log_p = F.log_softmax(model_noise_pred, dim=1)
                penalty = torch.sum(-p * log_p, dim=1).mean()

                return base_hint_noise + penalty_lambda * penalty
            else:
                raise ValueError(f"Unknown strategy {strategy}")

        # compute the mix weights
        hint_weight = get_mix_weight(timestep_idx, total_timesteps, gamma=2.0)
        # compute the hint noise
        hint_noise = get_mixed_noise(
            model_noise_pred=noise_pred,
            std_noise=std_noise,
            hint_weight=hint_weight,
            strategy="penalty",
        )

        return hint_noise

    def ddim_forward_step(
        pipe,
        x_t,
        custom_timesteps,
        prompt_embeds,
        encode_prompt_addtions,
        std_noise,
    ):
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
                # get hint noise
                hint_noise = get_hint_noise(noise_pred_c, std_noise, i+1, len(custom_timesteps))
                # ddim singe step
                result_c = ddim_single_step(hint_noise, x_c, t_c, t_target, pipe)
                x_c = result_c
                t_c = t_target

            # get target x_s
            x_s = x_c
        return x_s
    ################################################################################################################

    # get custom timesteps
    custom_timesteps = list(reversed(range(100, 200 + 1, 50)))

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

        # get x_t image
        x_t_latents, std_noise = get_init_x_t(pipe,
                                              x_0_latents,
                                              prompt_embeds,
                                              encode_prompt_addtions,
                                              torch.tensor(custom_timesteps[0]))

        tmp_forward_images = []
        for i in range(attack_args.num_samples):
            generator = torch.Generator("cuda").manual_seed(attack_args.global_seed + i)
            # forward
            x_s_latents = ddim_forward_step(
                pipe=pipe,
                x_t=x_t_latents,
                custom_timesteps=custom_timesteps,
                prompt_embeds=prompt_embeds,
                std_noise=std_noise,
                encode_prompt_addtions=encode_prompt_addtions,
            )

            # decode
            image_output = latent_decode(x_s_latents, pipe, generator)

            tmp_forward_images.append(pil2tensor(image_output))

        output_images.append(tmp_forward_images)
    # calculate similarity
    metric_scores = calculate_similarity(output_images, dataset_images, choice=choice)
    return metric_scores
