from torchvision import transforms as tsfm
import torch
from PIL import Image

def pil2tensor(pil_image):
    transform = tsfm.Compose(
        [tsfm.ToTensor(),
         tsfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize from pil to [-1, 1]
    tensor_image = transform(pil_image)
    return tensor_image


def tensor2pil(tensor_image):
    transform = tsfm.Compose(
        [tsfm.Normalize((-1., -1., -1.), (2., 2., 2.)), # denormalize from [-1, 1] to [0, 1]
         tsfm.ToPILImage()])
    pil_image = transform(tensor_image)
    return pil_image


def numpy_to_pil(image):
    """
    Convert a numpy image to a PIL image.
    """
    image = (image * 255).round().astype("uint8")
    if image.shape[-1] == 1:
        pil_image = Image.fromarray(image.squeeze(), mode="L")
    else:
        pil_image = Image.fromarray(image)

    return pil_image


def get_embeded_prompt(pipe, prompt):
    pipe_cls_name = pipe.__class__.__name__
    if pipe_cls_name in [
            "StableDiffusionPipeline",
            "StableDiffusionImg2ImgPipeline",
            "Kandinsky3Pipeline",
    ]:
        encode_prompt_fn = pipe.encode_prompt
    else:
        raise ValueError(
            f"TODO: pipe type {pipe_cls_name} is not supported")

    with torch.no_grad():
        prompt_embeds, *rest_ret = encode_prompt_fn(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
    encode_prompt_addtions = {}
    if pipe_cls_name in ["Kandinsky3Pipeline",]:
        encode_prompt_addtions.update({'encoder_attention_mask': rest_ret[1]})

    return prompt_embeds, encode_prompt_addtions


def get_encoded_latents(image_tensor, pipe):
    """
    Encode dataset images into latents using the given pipe.

    Returns:
        torch.Tensor: Encoded latents (B, C, H, W).
    """
    pipe_cls_name = pipe.__class__.__name__
    # make sure the VAE is in float32 mode, as it overflows in float16
    needs_upcasting = (pipe_cls_name == "StableDiffusionXLPipeline" and pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast)

    if needs_upcasting:
        pipe.upcast_vae()
        image_tensor = image_tensor.to(pipe.vae.dtype)

    # vae encoding to latents
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor.to('cuda')).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    # cast back to fp16 if needed
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)
        latents = latents.to(dtype=torch.float16)

    return latents


def latent_decode(latents, pipe, generator):
    """
    Decode latents into PIL image.
    """
    pipe_cls_name = pipe.__class__.__name__
    if pipe_cls_name in [
            "StableDiffusionPipeline",
            "StableDiffusionImg2ImgPipeline",
    ]:
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = (pipe_cls_name == "StableDiffusionXLPipeline" and pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast)

        if needs_upcasting:
            pipe.upcast_vae()
            latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

        image_t = pipe.vae.decode(latents / pipe.vae.config.scaling_factor,
                                  return_dict=False,
                                  generator=generator)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)

        image_t = pipe.image_processor.postprocess(image_t.detach(),
                                                   output_type="pil",
                                                   do_denormalize=[True] * image_t.shape[0])[0]


    elif pipe_cls_name == "Kandinsky3Pipeline":
        image_t = pipe.vae.decode(latents,
                                  return_dict=False,
                                  force_not_quantize=True)[0]

        # image_t = image_t * 0.5 + 0.5
        # image_t = image_t.clamp(0, 1)
        # image_t = image_t.cpu().permute(0, 2, 3, 1).float().numpy()

        # image_t = numpy_to_pil(image_t.squeeze())
        image_t = tensor2pil(image_t.squeeze())
    else:
        raise ValueError(f"TODO: pipe type {pipe_cls_name} is not supported")

    return image_t
