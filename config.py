import logging
import argparse
import sys
import torch
import random
from typing import List
import numpy as np
import datetime
import os


class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description='DPGUNet')
        parser.add_argument('--num_classes', default=4,
                            type=int, help='num of classes')
        parser.add_argument('--net_depth', default=4, type=int,
                            help='num of RGCMF, note that image conv layer not include!')
        parser.add_argument('--height', default=256,
                            type=int, help='height of image')
        parser.add_argument('--width', default=256,
                            type=int, help='width of image')
        parser.add_argument('--channels', default=1,
                            type=int, help='channels of image')

        # dataset
        parser.add_argument('--dataset_name', default='munich')
        parser.add_argument('--train_ratio', default=0.05, type=float)
        parser.add_argument('--val_ratio', default=0.01, type=float)

        # optimizer
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--lr', default=5e-4,
                            type=float, help='learning rate')

        # logging
        parser.add_argument('--save_map', type=bool, default=False,
                            help='whether save classification map')
        parser.add_argument('--work_dirs', default='./work_dirs',
                            help='exp log saved at work_dirs')
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--log_file_name', default='exp_log.txt')

        self.args = parser.parse_args()
        self.args.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if self.args.dataset_name == "munich":
            self.args.num_classes = 4
            self.args.image_dir = r'data\munich_s1\munich_s1'
            self.args.gt_dir = r'data\munich_s1\munich_anno'
            self.args.segments_dir = r'data\munich_s1\munich_segments'
        else: # todo add other datasets
            raise ValueError
        self._set_seed(self.args.seed)
        self._configure_logger()
        self._print_args()

    def get_args(self):
        return self.args

    def _print_args(self):
        self.args.logger.info(
            "*******************       args      *******************")
        for arg, content in self.args.__dict__.items():
            self.args.logger.info("{}:{}".format(arg, content))
        self.args.logger.info(
            "*******************     args END    *******************")

    def _configure_logger(self):
        logger = logging.getLogger(name='exp-log')
        logger.setLevel('DEBUG')

        date_time = datetime.datetime.now()
        time = date_time.strftime("%Y%m%d_%H%M%S")
        # print(time)
        self.args.time = time
        self.args.work_dirs = os.path.join(self.args.work_dirs, self.args.dataset_name+"_"+time)
        os.makedirs(self.args.work_dirs)
        log_file = os.path.join(
            self.args.work_dirs, self.args.log_file_name)
        file_handler = logging.FileHandler(log_file)
        stdout_handler = logging.StreamHandler(sys.stdout)
        # formatter = logging.Formatter(
        #     fmt='%(asctime)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            fmt='%(asctime)s| %(message)s')
        file_handler.setLevel('INFO')
        file_handler.setFormatter(formatter)
        stdout_handler.setLevel('DEBUG')
        stdout_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

        logger.info('@'*88)
        logger.info('exp log file saved at {}'.format(log_file))
        self.args.logger = logger

    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
