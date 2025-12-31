def pipeline_sd_v1p5(base_model_id, PIPE_TYPE):
    """ 
    we use the custom diffusers (under custom_diffusers/ directory) provided by Duan et al. to excute their SecMI:
    Are Diffusion Models Vulnerable to Membership Inference Attacks? Duan, J., Kong, F., Wang, S., Shi, X., & Xu, K. (2023).  
    https://proceedings.mlr.press/v202/duan23b.html
    """
    # from custom_diffusers import StableDiffusionPipeline, DDIMScheduler
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    import torch
    # prepare model pipeline
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id,
                                                   safety_checker=None,
                                                   torch_dtype=PIPE_TYPE,
                                                   variant="fp16" if PIPE_TYPE == torch.float16 else None)
    # reset scheduler to DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    return pipe

def pipeline_commoncanvas(base_model_id, PIPE_TYPE):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    import torch
    # prepare model pipeline
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id,
                                                   safety_checker=None,
                                                   torch_dtype=PIPE_TYPE,
                                                   variant="fp16" if PIPE_TYPE == torch.float16 else None)
    # reset scheduler to DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    return pipe


def pipeline_sd_xl(base_model_id, PIPE_TYPE):
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler
    import torch
    # prepare model pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id,
                                                     safety_checker=None,
                                                     torch_dtype=PIPE_TYPE,
                                                     variant="fp16" if PIPE_TYPE == torch.float16 else None)
    # reset scheduler to DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')

    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    # pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    return pipe



def pipeline_kandinsky(base_model_id, PIPE_TYPE):
    from diffusers import Kandinsky3Pipeline, DDIMScheduler
    import torch
    pipe = Kandinsky3Pipeline.from_pretrained(base_model_id,
                                              torch_dtype=PIPE_TYPE,
                                              variant="fp16" if PIPE_TYPE == torch.float16 else None
                                              ).to('cuda')
    # reset scheduler to DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')

    # Rename movq to vae for unified interface
    class VQAdapter:
        def __init__(self, movq):
            self.movq = movq
            self.config = movq.config

        def encode(self, x: torch.FloatTensor, return_dict: bool = True):
            # call movq.encode
            result = self.movq.encode(x, return_dict=True)

            # create a vae-like interface for the latent distribution
            class LatentDistAdapter:
                def __init__(self, latents):
                    self.latents = latents

                def sample(self):
                    # do nothing
                    return self.latents

            # add latent_dist attribute
            if return_dict:
                result.latent_dist = LatentDistAdapter(result.latents)
            return result

        def decode(self, latents, return_dict=True, **kwargs):
            # call movq.decode
            result = self.movq.decode(latents, return_dict=return_dict, **kwargs)
            return result
    # rename movq to vae for unified interface
    pipe.vae = VQAdapter(pipe.movq)
    # remove clip_sample for kandinsky 3
    if pipe.scheduler.config.clip_sample:
        pipe.scheduler.config.clip_sample = False
    # do not scale the latents for kandinsky 3
    pipe.vae.config.scaling_factor = 1.0

    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe



def txt2img_pipeline_select(base_model_id, PIPE_TYPE):
    """
    General function to select pipeline based on model name.
    """
    if "stable-diffusion-v1p5" in base_model_id:
        return pipeline_sd_v1p5(base_model_id, PIPE_TYPE)
    elif "commoncanvas-s-c" in base_model_id:
        return pipeline_commoncanvas(base_model_id, PIPE_TYPE)
    elif "kandinsky-3" in base_model_id:
        return pipeline_kandinsky(base_model_id, PIPE_TYPE)
    else:
        raise ValueError(f"Model {base_model_id} not supported.")
