"""
Select different pipeline for different diffusion models.
"""

def pipeline_auto(base_model_id, PIPE_TYPE):
    from diffusers import AutoPipelineForImage2Image, DDIMScheduler

    # prepare model pipeline
    pipe = AutoPipelineForImage2Image.from_pretrained(
        base_model_id,
        safety_checker=None,
        torch_dtype=PIPE_TYPE,
        use_safetensors=True,
    ).to("cuda")

    # customize scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # optimization
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    return pipe


def img2img_pipeline_select(base_model_id, PIPE_TYPE):
    """
    General function to select pipeline based on model name.
    """
    return pipeline_auto(base_model_id, PIPE_TYPE)
