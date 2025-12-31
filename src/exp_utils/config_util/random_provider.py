import random

from exp_utils.config_util.attack_config import attack_args

# create global random
global_random = random.Random()
# set seed
global_random.seed(attack_args.global_seed)
