import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--model_type', type=str, default='ConvNet', choices=['ConvNet', 'ResNet', 'AmdimNet'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--gpu', default='0')

    parser.add_argument('--ndf', type=int, default=256)
    parser.add_argument('--rkhs', type=int, default=2048)
    parser.add_argument('--nd', type=int, default=10)


    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type, 'ProtoNet'])
    save_path2 = '_'.join([str(args.shot), str(args.query), str(args.way), 
                               str(args.step_size), str(args.gamma), str(args.lr), str(args.temperature)])
    args.save_path = osp.join(args.save_path, osp.join(save_path1, save_path2))
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path)