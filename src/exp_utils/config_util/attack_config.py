import argparse
import os
import json
import torch


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Pipeline group
    pipeline_group = parser.add_argument_group("Pipeline Config")
    pipeline_group.add_argument("--dtype",
                                default="float16",
                                choices=["float16", "float32"],
                                help="Pipeline data type")

    pipeline_group.add_argument("--base_model_id",
                                default="YOUR_MODEL_PATH",
                                help="Diffusion Model path")

    pipeline_group.add_argument("--local_dataset_path",
                                default="YOUR_LOCAL_DATASET_PATH",
                                help="Local dataset path, should contain all datasets used in this attack")

    pipeline_group.add_argument("--model_name",
                                type=str,
                                default="stable-diffusion-v1p5",
                                choices=[
                                    "stable-diffusion-v1p5",
                                    "commoncanvas-s-c",
                                    "kandinsky-3",
                                ],
                                help="Diffusion Model name")


    # Attack group
    attack_group = parser.add_argument_group("Attack Config")
    attack_group.add_argument("--num_samples",
                              type=int,
                              default=1,
                              help="Generate sample number")

    attack_group.add_argument("--generate_resolution",
                              type=int,
                              default=512,
                              help="Resolution of generation")

    # Dataset group
    dataset_group = parser.add_argument_group("Dataset Config")
    dataset_group.add_argument("--splits",
                               default=["train", "val"],
                               choices=["train", "val"],
                               nargs="+",
                               help="Dataset split")
    dataset_group.add_argument(
        "--used_datasets",
        default=[
            "laion_aesthetics_v2_5p0",
            "D_hold_out_laion_aesthetics_v2_5p0",
        ],
        choices=[
            ## non-iid datasets
            # training sets
            "commoncatalog_cc_by_sa",
            "laion_aesthetics_v2_5p0",
            # hold-out source data
            "coco_2017_val",
            "laion2b_multi_translated",
            "nocaps",
            # calibrated hold-out sets
            "D_base_1",
            "D_base_2",
            
            # TPR group hold-out sets
            "D_hold_out_commoncatalog_cc_by_sa",
            "D_hold_out_laion_aesthetics_v2_5p0",

            # FPR group hold-out sets
            "D_hold_out_1",
            "D_hold_out_2",
        ],
        nargs="+",
        help="Used datasets")

    dataset_group.add_argument(
        "--global_seed",
        type=int,
        default=12345,  # int(time.time())
        help="Load seed")

    dataset_group.add_argument("--dataset_total_num",
                               type=int,
                               default=1000,
                               help="Total number of dataset")

    # baseline group
    baseline_group = parser.add_argument_group("Baseline Config")
    baseline_group.add_argument(
        "--baseline",
        type=str,
        default="structuralmi",
        choices=["secmi", "structuralmi", "naivelossmi", "piami", "rediffusemi", "gradient_maskingmi", "multiple_lossmi", "noise_optimmi"],
        help="baseline type")

    # Debug group
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "--is_visualize",
        action="store_true",
        default=False,
        help="Visualize mode, will save timestep images to output directory")
    debug_group.add_argument(
        "--is_remove",
        action="store_false",
        default=True,
        help="Remove mode, will remove all output files"
    )

    return parser


# Global configuration
print("ArgumentParser Initializing...")

# create attack args
parser = create_parser()
attack_args = parser.parse_args()
# datesetname for plot
dataset_plot_names = {
    "coco_2017_val": "COCO2017-val",
    "laion_aesthetics_v2_5p0": "LAION Aesthetics v2 5+",
    "commoncatalog_cc_by_sa": "CommonCatalog CC-BY-SA",
    "laion2b_multi_translated": "LAION-2B Multi Translated",
    "nocaps": "NoCaps",
    "D_hold_out_commoncatalog_cc_by_sa": "Hold-out of CommonCatalog CC-BY-SA",
    "D_hold_out_laion_aesthetics_v2_5p0": "Hold-out of LAION Aesthetics v2 5+",
    "D_base_1": "Calibrated Hold-out Base-1",
    "D_base_2": "Calibrated Hold-out Base-2",
}
model_plot_names = {
    "stable-diffusion-v1p5": "Stable-Diffusion-v1-5",
    "commoncanvas-s-nc": "CommonCanvas-S-NC",
    "kandinsky-3": "Kandinsky-3",
}
# get model path
if attack_args.dtype == "float16":
    attack_args.base_model_id = os.path.join(
        attack_args.base_model_id, attack_args.model_name, "fp16")
    print(f"Using {attack_args.base_model_id}")
else:
    attack_args.base_model_id = os.path.join(
        attack_args.base_model_id, attack_args.model_name, "fp32")
    print(f"Using {attack_args.base_model_id}")
# set pipeline dtype
PIPE_TYPE = torch.float32 if attack_args.dtype == "float32" else torch.float16
# check args legality
assert len(attack_args.used_datasets) == 2, \
    "attack_args.used_datasets must have two elements."
assert len(attack_args.splits) == 2, \
    "attack_args.splits must have two elements."


# output root path
OUTPUT_ROOT_PATH = f"./exp_data"
# statistic path
STATISTIC_ROOT_PATH = f"{OUTPUT_ROOT_PATH}/statistic/"
# plot path
PLOT_DIR = f"{STATISTIC_ROOT_PATH}/plot/"
# MI evaluation path
MI_DIR = f"{OUTPUT_ROOT_PATH}/mi_eval/"

############################################## train set ##############################################
# create dataset config
dataset_config_1 = {
    "dataset": attack_args.used_datasets[0],
    "load_path": f"{attack_args.local_dataset_path}/{attack_args.used_datasets[0]}/data/{attack_args.splits[0]}",
    "load_num": attack_args.dataset_total_num,
    "split": attack_args.splits[0],
}

# create paths
# output path
TRAIN_OUTPUT_DIR = f"{OUTPUT_ROOT_PATH}/output/{dataset_config_1['dataset']}/{attack_args.model_name}/"
# statistic path
TRAIN_STATISTIC_DIR = f"{STATISTIC_ROOT_PATH}/{dataset_config_1['dataset']}/{attack_args.model_name}/"

###################################################### hold out set #############################################
# create dataset config
dataset_config_2 = {
    "dataset": attack_args.used_datasets[1],
    "load_path": f"{attack_args.local_dataset_path}/{attack_args.used_datasets[1]}/data/{attack_args.splits[1]}",
    "load_num": attack_args.dataset_total_num,
    "split": attack_args.splits[1],
}

# create paths
# output path
HOLD_OUT_OUTPUT_DIR = f"{OUTPUT_ROOT_PATH}/output/{dataset_config_2['dataset']}/{attack_args.model_name}/"
# statistic path
HOLD_OUT_STATISTIC_DIR = f"{STATISTIC_ROOT_PATH}/{dataset_config_2['dataset']}/{attack_args.model_name}/"


##################################################### save arguments configuration #####################################
# save arguments configuration
os.makedirs(OUTPUT_ROOT_PATH, exist_ok=True)
with open(
        f"{OUTPUT_ROOT_PATH}/{attack_args.used_datasets[0]}_{attack_args.used_datasets[1]}_{attack_args.model_name}_args_config.json",
        "w",
        encoding="utf-8") as f:
    f.write(
        json.dumps(
            {
                "attack_args": vars(attack_args),
                "dataset_config_1": dataset_config_1,
                "train_output_dir": TRAIN_OUTPUT_DIR,
                "train_statistic_dir": TRAIN_STATISTIC_DIR,
                "dataset_config_2": dataset_config_2,
                "hold_out_output_dir": HOLD_OUT_OUTPUT_DIR,
                "hold_out_statistic_dir": HOLD_OUT_STATISTIC_DIR,
                "mi_eval_dir": MI_DIR,
            },
            indent=4,
            ensure_ascii=False))


####################################### Pipeline Inference Related #############################################
# Unified parameter configuration
def get_unet_kwargs(pipe, **kwargs):
    # update kwargs
    encode_prompt_addtions = kwargs.get("encode_prompt_addtions", None)
    kwargs.update(encode_prompt_addtions)

    # get unet_kwargs according to pipe type
    pipe_cls_name = pipe.__class__.__name__
    unet_kwargs = {}

    if pipe_cls_name in [
            "StableDiffusionPipeline",
            "StableDiffusionImg2ImgPipeline",
    ]:
        unet_kwargs = {
            'sample': kwargs.get("sample", None),
            'timestep': kwargs.get("timestep", None),
            'encoder_hidden_states':kwargs.get("encoder_hidden_states", None),
            'return_dict':kwargs.get("return_dict", False),
        }
    elif pipe_cls_name in [
        "Kandinsky3Pipeline",
    ]:
        unet_kwargs = {
            'sample': kwargs.get("sample", None),
            'timestep': kwargs.get("timestep", None),
            'encoder_hidden_states':kwargs.get("encoder_hidden_states", None),
            'encoder_attention_mask':kwargs.get("encoder_attention_mask", None),
            'return_dict':kwargs.get("return_dict", False),
        }
    elif pipe_cls_name in [
            "StableDiffusionXLPipeline",
    ]:
        # get prompt_embeds
        prompt_embeds = kwargs.get("encoder_hidden_states", None)
        pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds", None)
        batch_size = prompt_embeds.shape[0]

        # prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

        add_time_ids = pipe._get_add_time_ids(
            (attack_args.generate_resolution, attack_args.generate_resolution),
            (0, 0),
            (attack_args.generate_resolution, attack_args.generate_resolution),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        prompt_embeds = prompt_embeds.to(pipe._execution_device)
        add_text_embeds = add_text_embeds.to(pipe._execution_device)
        add_time_ids = add_time_ids.to(pipe._execution_device).repeat(batch_size * 1, 1)

        # unet_kwargs
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        unet_kwargs = {
            'sample': kwargs.get("sample", None),
            'timestep': kwargs.get("timestep", None),
            'encoder_hidden_states':prompt_embeds,
            'added_cond_kwargs':added_cond_kwargs,
            'return_dict':kwargs.get("return_dict", False),
        }

    return unet_kwargs
