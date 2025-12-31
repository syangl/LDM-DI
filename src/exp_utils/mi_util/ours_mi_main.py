import os, sys, shutil
# get the current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# get the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_dir))
# # add the project root directory as the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from exp_utils.config_util.attack_config import attack_args, TRAIN_OUTPUT_DIR, dataset_config_1, HOLD_OUT_OUTPUT_DIR, dataset_config_2, PIPE_TYPE, MI_DIR
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.mi_util.ours_hintmi_src.ours_hintmi import execute_hintmi
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select
from exp_utils.metric_util.evaluate import evaluate_roc
import json




############################################### Entrance #########################################

def inference(split):
    print()
    print(f"{split} Inference...")
    print()
    if split == "train":
        OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, "ours_hintmi")
        dataset_config = dataset_config_1
    elif split == "hold_out":
        OUTPUT_DIR = os.path.join(HOLD_OUT_OUTPUT_DIR, "ours_hintmi")
        dataset_config = dataset_config_2
    else:
        raise ValueError(f"Unknown split: {split}")


    # Inference
    # generate images
    if os.path.exists(OUTPUT_DIR):
        print(f"Output already exist in {OUTPUT_DIR}, skipping generation.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # prepare model pipeline
        pipe_txt2img = txt2img_pipeline_select(attack_args.base_model_id, PIPE_TYPE)
        # bypass tensor input check
        # bypass_callback_tensor_inputs_check(pipe_txt2img)

        # load dataset images
        print(f"loading image...")
        dataloader, _ = create_dataset_dataloader(
            OUTPUT_DIR,
            dataset_config,
            pipe_txt2img.tokenizer,
            img_resolution=attack_args.generate_resolution,
            batch_size=1,
            need_save=attack_args.is_visualize,
        )

        # Hinting denoising
        print(f"Hinting denoising...")
        print()
        scores = execute_hintmi(dataloader=dataloader, pipe=pipe_txt2img)

        return scores


if __name__ == '__main__':
    member_scores_file = os.path.join(TRAIN_OUTPUT_DIR, "ours_hintmi", "hintmi_member_scores.json")
    nonmember_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, "ours_hintmi", "hintmi_nonmember_scores.json")
    if attack_args.is_remove:
        # remove old files
        shutil.rmtree(os.path.join(TRAIN_OUTPUT_DIR, "ours_hintmi"), ignore_errors=True)
        shutil.rmtree(os.path.join(HOLD_OUT_OUTPUT_DIR, "ours_hintmi"), ignore_errors=True)
        # inference
        member_scores = inference("train")
        nonmember_scores = inference("hold_out")

        # save results
        with open(member_scores_file, "w", encoding='utf-8') as f:
            json.dump({"member_scores": member_scores}, f)

        with open(nonmember_scores_file, "w", encoding='utf-8') as f:
            json.dump({"nonmember_scores": nonmember_scores}, f)
    else:
        member_scores = json.load(open(member_scores_file, "r")).values()
        nonmember_scores = json.load(open(nonmember_scores_file, "r")).values()

    # MI evaluation
    res_dir = os.path.join(MI_DIR, "ours_hintmi")
    evaluate_roc(member_scores, nonmember_scores, save_path=res_dir, larger_is_member=True)
