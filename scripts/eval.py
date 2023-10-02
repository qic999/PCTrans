import os
import subprocess
import sys
sys.path.append(os.getcwd())
from connectomics.inference.evaluation.evaluate_epfl import eval_epfl
# from connectomics.inference.evaluation.evaluate_kar import eval_kar
from connectomics.inference.evaluation.evaluate_mito import eval_mito
from connectomics.inference.evaluation.evaluate_snemi import eval_snemi
from connectomics.inference.evaluation.evaluate_snemi2d import eval_snemi2d
from connectomics.config.defaults import get_cfg_defaults
from connectomics.config.maskfoermer_config import add_maskformer2_config
import argparse
from tensorboardX import SummaryWriter
import time

def cal_infer_bbbc(model_dir, model_id, config_base, config_file):
    command = "python scripts/main.py --config-base\
            {}\
            --config-file\
            {}\
            --inference\
            --checkpoint\
            {}checkpoint_{:06d}.pth.tar\
            --opts\
            SYSTEM.NUM_GPUS\
            1\
            SYSTEM.NUM_CPUS\
            3\
            INFERENCE.SAMPLES_PER_BATCH\
            2\
            INFERENCE.INPUT_SIZE\
            [1,520,696]\
            INFERENCE.OUTPUT_SIZE\
            [1,520,696]\
            INFERENCE.STRIDE\
            [1,1,1]\
            INFERENCE.PAD_SIZE\
            [0,0,0]\
            INFERENCE.AUG_NUM\
            None\
        ".format(config_base, config_file, model_dir, model_id)
    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")


def cal_infer_cvppp(model_dir, model_id, config_base, config_file):
    command = "python scripts/main.py --config-base\
            {}\
            --config-file\
            {}\
            --inference\
            --checkpoint\
            {}checkpoint_{:06d}.pth.tar\
            --opts\
            SYSTEM.NUM_GPUS\
            1\
            SYSTEM.NUM_CPUS\
            3\
            INFERENCE.SAMPLES_PER_BATCH\
            2\
            INFERENCE.INPUT_SIZE\
            [1,1024,1024]\
            INFERENCE.OUTPUT_SIZE\
            [1,1024,1024]\
            INFERENCE.STRIDE\
            [1,1,1]\
            INFERENCE.PAD_SIZE\
            [0,0,0]\
            INFERENCE.AUG_NUM\
            None\
        ".format(config_base, config_file, model_dir, model_id)
    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")


def get_args():
    r"""Get args from command lines.
    """
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)
    parser.add_argument('--test_model_list', nargs='+', type=int)
    parser.add_argument('--name', type=str, default='snemi')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    # print(sys.path)
    args = get_args()
    cfg = get_cfg_defaults()
    add_maskformer2_config(cfg)
    if args.config_base is not None:
        cfg.merge_from_file(args.config_base)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    start_iter = 51000
    end_iter = cfg.SOLVER.ITERATION_TOTAL
    step_iter = cfg.SOLVER.ITERATION_SAVE
    model_ids = range(start_iter, end_iter+step_iter, step_iter)

    model_dir=cfg.DATASET.OUTPUT_PATH
    pre_dir=cfg.INFERENCE.OUTPUT_PATH
    name = args.name
    record_path = os.path.join(pre_dir, name)
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    eval_writer = SummaryWriter(record_path)
    start_time = time.time()
    for model_id in model_ids:
        if args.name == 'bbbc':
            score = cal_infer_bbbc(model_dir, model_id, args.config_base, args.config_file)
        elif args.name == 'cvppp':
            cal_infer_cvppp(model_dir, model_id, args.config_base, args.config_file)

    end_time = time.time()
    day = (end_time - start_time) // (24*60*60)
    hour = (end_time - start_time - day*(24*60*60)) // (60*60)
    minu = (end_time - start_time - day*(24*60*60) - hour*(60*60)) // 60
    total_time = print(f"{day}day {hour}hour {minu}min")
    print('total_time:', total_time)



    