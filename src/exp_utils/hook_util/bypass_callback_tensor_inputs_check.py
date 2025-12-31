

def bypass_callback_tensor_inputs_check(pipeline):
    """
    Bypass the callback tensor inputs check of the pipeline to allow more internal variables to be passed to the callback.
    This is necessary because we need to add noise to the latents in the callback.
    """
    # save the original callback tensor inputs
    original_callback_tensor_inputs = pipeline._callback_tensor_inputs

    # change the callback tensor inputs to include latents, prompt_embeds, negative_prompt_embeds, noise_pred, and extra_step_kwargs
    pipeline._callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "noise_pred",  # add noise_pred
        "extra_step_kwargs"  # add extra_step_kwargs
    ]
