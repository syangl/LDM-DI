"""
Baseline attacks in our experiments
"""
import sys
import os, shutil

# get the current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# get the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_dir))
# add the project root directory as the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_utils.config_util.attack_config import TRAIN_OUTPUT_DIR, HOLD_OUT_OUTPUT_DIR
from exp_utils.config_util.attack_config import attack_args



def execute_secmi():
    from exp_utils.mi_util.secmi_src.secmi import fix_seed, secmi_main

    fix_seed(attack_args.global_seed)

    secmi_main()


def execute_structuralmia():
    from exp_utils.mi_util.structuralmi_src.structuralmi import structuralmi_main

    structuralmi_main()

def execute_piami():
    from exp_utils.mi_util.piami_src.piami import piami_main

    piami_main()

def execute_naivelossmi():
    from exp_utils.mi_util.naivelossmi_src.naivelossmi import naivelossmi_main

    naivelossmi_main()


def execute_rediffusemi():
    from exp_utils.mi_util.rediffusemi_src.rediffusemi import rediffusemi_main

    rediffusemi_main()



if __name__ == "__main__":
    # for debug
    if attack_args.is_remove:
        # remove old files
        shutil.rmtree(os.path.join(TRAIN_OUTPUT_DIR, attack_args.baseline), ignore_errors=True)
        shutil.rmtree(os.path.join(HOLD_OUT_OUTPUT_DIR, attack_args.baseline), ignore_errors=True)

    # baselines
    if attack_args.baseline == "secmi":
        # add the src_paths directory to the Python path
        src_path = os.path.join(project_root, "exp_utils/mi_util/secmi_src") # we use SecMI customized diffuers module when executing SecMI
        sys.path.insert(0, src_path)
        # execute SecMI
        execute_secmi()
    elif attack_args.baseline == "structuralmi":
        # add the src_paths directory to the Python path
        src_path = os.path.join(project_root, "exp_utils/mi_util/structuralmi_src")
        sys.path.insert(0, src_path)
        # execute StructuralMI
        execute_structuralmia()
    elif attack_args.baseline == "piami":
        # add the src_paths directory to the Python path
        src_path = os.path.join(project_root, "exp_utils/mi_util/piami_src")
        sys.path.insert(0, src_path)
        # execute PIAMI
        execute_piami()
    elif attack_args.baseline == "naivelossmi":
        # add the src_paths directory to the Python path
        src_path = os.path.join(project_root, "exp_utils/mi_util/naivelossmi_src")
        sys.path.insert(0, src_path)
        # execute NaiveLossMI
        execute_naivelossmi()
    elif attack_args.baseline == "rediffusemi":
        # add the src_paths directory to the Python path
        src_path = os.path.join(project_root, "exp_utils/mi_util/rediffusemi_src")
        sys.path.insert(0, src_path)
        # execute RediffuseMI
        execute_rediffusemi()
    else:
        raise ValueError("Invalid baseline type")
