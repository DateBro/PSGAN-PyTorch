import os

import argparse

from torch.backends import cudnn
from config import config, dataset_config, merge_cfg_arg

from dataloder import get_loader
from solver_psgan import Solver_PSGAN

def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN')
    # general
    # parser.add_argument('--data_path', default='F:/zzy/data/makeup_dataset', type=str, help='training and test data path')
    parser.add_argument('--data_path', default='/home/zhiyong/RemoteServer/data/makeup_dataset', type=str, help='training and test data path')
    # parser.add_argument('--data_path', default='makeup/makeup_final/', type=str, help='training and test data path')
    parser.add_argument('--dataset', default='MAKEUP', type=str, help='dataset name, MAKEUP means two domain, MMAKEUP means multi-domain')
    parser.add_argument('--gpus', default='0', type=str, help='GPU device to train with')
    parser.add_argument('--batch_size', default='1', type=int, help='batch_size')
    parser.add_argument('--vis_step', default='1260', type=int, help='steps between visualization')
    parser.add_argument('--task_name', default='', type=str, help='task name')
    parser.add_argument('--checkpoint', default='', type=str, help='checkpoint to load')
    parser.add_argument('--ndis', default='1', type=int, help='train discriminator steps')
    parser.add_argument('--LR', default="2e-4", type=float, help='Learning rate')
    parser.add_argument('--decay', default='0', type=int, help='epochs number for training')
    parser.add_argument('--model', default='PSGAN', type=str, help='which model to use: cycleGAN/ makeupGAN')
    parser.add_argument('--epochs', default='50', type=int, help='nums of epochs')
    parser.add_argument('--norm', default='SN', type=str, help='normalization of discriminator, SN means spectrum normalization, none means no normalization')
    parser.add_argument('--d_repeat', default='3', type=int, help='the repeat Res-block in discriminator')
    parser.add_argument('--g_repeat', default='6', type=int, help='the repeat Res-block in Generator')
    parser.add_argument('--lambda_cls', default='1', type=float, help='the lambda_cls weight')
    parser.add_argument('--lambda_rec', default='10', type=int, help='lambda_A and lambda_B')
    parser.add_argument('--lambda_his', default='1', type=float, help='histogram loss on lips')
    parser.add_argument('--lambda_skin_1', default='0.1', type=float, help='histogram loss on skin equals to lambda_his* lambda_skin')
    parser.add_argument('--lambda_skin_2', default='0.1', type=float, help='histogram loss on skin equals to lambda_his* lambda_skin')
    parser.add_argument('--lambda_eye', default='1', type=float, help='histogram loss on eyes equals to lambda_his*lambda_eye')
    parser.add_argument('--content_layer', default='r41', type=str, help='vgg layer we use to output features')
    parser.add_argument('--lambda_vgg', default='5e-3', type=float, help='the param of vgg loss')
    # SYMIX是non-makeup，MAKEMIX是makeup
    parser.add_argument('--cls_list', default='SYMIX,MAKEMIX', type=str, help='the classes of makeup to train')
    parser.add_argument('--direct', action="store_true", default=True, help='direct means to add local cosmetic loss at the first, unified training')
    parser.add_argument('--lips', action="store_true", default=True, help='whether to finetune lips color')
    parser.add_argument('--skin', action="store_true", default=True, help='whether to finetune foundation color')
    parser.add_argument('--eye', action="store_true", default=True, help='whether to finetune eye shadow color')
    args = parser.parse_args()
    return args

def train_net():
    # enable cudnn
    cudnn.benchmark = True
    data_loaders = get_loader(dataset_config, config, mode="train")
    solver = Solver_PSGAN(data_loaders, config, dataset_config)
    solver.train()

if __name__ == '__main__':
    args = parse_args()
    print("Call with args:")
    print(args)
    config = merge_cfg_arg(config, args)

    dataset_config.name = args.dataset

    print("The config is:")
    print(config)

    # Create the directories if not exist
    if not os.path.exists(config.data_path):
        print("No datapath!!")
        exit()

    if args.data_path != '':
        dataset_config.dataset_path = os.path.join(config.data_path, args.data_path)

    train_net()
