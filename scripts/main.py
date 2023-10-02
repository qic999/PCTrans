import os
import torch
import os, sys
sys.path.append(os.getcwd())
import time

from connectomics.utils.system import get_args, init_devices
from connectomics.config import load_cfg, save_all_cfg
from connectomics.engine import Trainer


def main():
    start_time = time.time()
    args = get_args()
    cfg = load_cfg(args)
    device = init_devices(args, cfg)

    if args.local_rank == 0 or args.local_rank is None:
        # In distributed training, only print and save the configurations 
        # using the node with local_rank=0.
        print("PyTorch: ", torch.__version__)
        print(cfg)

        if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
            print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
            os.makedirs(cfg.DATASET.OUTPUT_PATH)
            save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    # start training or inference
    mode = 'test' if args.inference else 'train'
    trainer = Trainer(cfg, device, mode,
                      rank=args.local_rank,
                      checkpoint=args.checkpoint)

    # Start training or inference:
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        if cfg.DATASET.DATA_TYPE == 'CVPPP':
            test_func = trainer.eval_cvppp
            # test_func = trainer.test_cvppp
        elif cfg.DATASET.DATA_TYPE == 'BBBC':
            test_func = trainer.test_bbbc
        elif cfg.DATASET.DATA_TYPE == 'monuseg':
            test_func = trainer.test_monuseg
        elif cfg.DATASET.DATA_TYPE == 'cellpose':
            test_func = trainer.test_cellpose
        else:
            if cfg.INFERENCE.DO_SINGLY:
                test_func = trainer.test_singly  
            else:
                if cfg.MODEL.TARGET_OPT == ['9']:
                    test_func = trainer.test_inst
                else:
                    test_func = trainer.test
        test_func() if args.inference else trainer.train()
    else:
        trainer.run_chunk(mode)

    print("Rank: {}. Device: {}. Process is finished!".format(
          args.local_rank, device))
          
    end_time = time.time()
    day = (end_time - start_time) // (24*60*60)
    hour = (end_time - start_time - day*(24*60*60)) // (60*60)
    min = (end_time - start_time - day*(24*60*60) - hour*(60*60)) // 60
    print(f"{day}day {hour}hour {min}min")
    
    


if __name__ == "__main__":
    main()
