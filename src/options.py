#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from inspect import getmembers, isfunction, isclass
from ast import literal_eval
from datetime import datetime
import sys
from torch.cuda import device_count
from utils import optimizers, schedulers


def args_parser():
    #max_help_position=1000, width=1000
    usage = 'python main.py [ARGUMENTS]'
    parser = argparse.ArgumentParser(prog='main.py', usage=usage, add_help=False, formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog))

    # Algorithm arguments
    args_algo = parser.add_argument_group('algorithm arguments')

    args_algo_rounds_iters = args_algo.add_mutually_exclusive_group()
    args_algo_rounds_iters.add_argument('--rounds', type=int, default=500,
                        help="number of communication rounds, or number of epochs if --centralized")
    args_algo_rounds_iters.add_argument('--iters', type=int, default=None,
                        help="number of iterations: the iterations of a round are determined by the client with the largest number of images")
    args_algo.add_argument('--num_clients', '-K', type=int, default=6,
                        help="number of clients")
    args_algo.add_argument('--frac_clients', '-C', type=float, default=1,
                        help="fraction of clients selected at each round")
    args_algo.add_argument('--train_bs', '-B', type=int, default=2,
                        help="client training batch size, 0 to use the whole training set")
    args_algo.add_argument('--epochs', '-E', type=int, default=5,
                        help="number of client epochs")
    args_algo.add_argument('--hetero', type=float, default=0,
                        help="probability of clients being stragglers, i.e. training for less than EPOCHS epochs")
    # args_algo.add_argument('--drop_stragglers', action='store_true', default=False,
    #                     help="drop stragglers")
    args_algo.add_argument('--server_lr', type=float, default=1,
                        help="server learning rate")
    args_algo.add_argument('--server_momentum', type=float, default=0,
                        help="server momentum for FedAvgM algorithm")#0.7
    args_algo.add_argument('--mu', type=float, default=0,
                        help="mu parameter for FedProx algorithm")#0.2
    args_algo.add_argument('--centralized', action='store_true', default=False,
                        help="use centralized algorithm")
    args_algo.add_argument('--usepro', type=bool, default=True,
                        help="use prototypes")
    args_algo.add_argument('--dataset', type=str, default="glm",
                        help="dataset")
    args_algo.add_argument('--lambda0', type=float, default=1,
                        help="lambda1")
    args_algo.add_argument('--lambda1', type=float, default=0.01,
                        help="lambda1")
    args_algo.add_argument('--lambda2', type=float, default=0.01,
                        help="lambda2")
    args_algo.add_argument('--algorithm', type=str, default="FedPARS",
                           help="algorithm")
    args_algo.add_argument('--num_prototype', type=int, default=20,
                           help="num_prototype")
    args_algo.add_argument('--gamma_list',nargs='*', type=float, default=[0.5,0.5],
                           help="num_prototype")

    # Model, optimizer and scheduler arguments
    args_model_optim_sched = parser.add_argument_group('model, optimizer and scheduler arguments')
    # args_model_optim_sched.add_argument('--model', type=str, default='lenet5', choices=[c[0] for c in getmembers(models, isclass) if c[1].__module__ == 'models'],
    #                     help="model, place yours in models.py")
    # args_model_optim_sched.add_argument('--model_args', type=str, default='ghost=True,norm=None',
    #                     help="model arguments")
    args_model_optim_sched.add_argument('--optim', type=str, default='sgd', choices=[f[0] for f in getmembers(optimizers, isfunction)],
                        help="optimizer, place yours in optimizers.py")
    args_model_optim_sched.add_argument('--optim_args', type=str, default='lr=0.005,momentum=0,weight_decay=4e-4',
                        help="optimizer arguments")
    args_model_optim_sched.add_argument('--sched', type=str, default='fixed', choices=[c[0] for c in getmembers(schedulers, isclass) if c[1].__module__ == 'schedulers'],
                        help="scheduler, place yours in schedulers.py")
    args_model_optim_sched.add_argument('--sched_args', type=str, default=None,
                        help="scheduler arguments")

    # Output arguments
    args_output = parser.add_argument_group('output arguments')
    args_output.add_argument('--client_stats_every', type=int, default=0,
                        help="compute and print client statistics every CLIENT_STATS_EVERY batches, 0 for every epoch")
    args_output.add_argument('--server_stats_every', type=int, default=1,
                        help="compute, print and log server statistics every SERVER_STATS_EVERY rounds")
    args_output.add_argument('--name', type=str, default='testfed',
                        help="log to runs/NAME and save checkpoints to save/NAME, None for YYYY-MM-DD_HH-MM-SS")
    args_output.add_argument('--no_log', action='store_true', default=False,
                        help="disable logging")
    args_output.add_argument('--no_save', action='store_true', default=False,
                        help="disable checkpoints")
    args_output.add_argument('--quiet', '-q', action='store_true', default=False,
                        help="less verbose output")

    # Other arguments
    args_other = parser.add_argument_group('other arguments')
    args_other.add_argument('--test_bs', type=int, default=256,
                        help="client valid/validation batch size")
    args_other.add_argument('--seed', type=int, default=0,
                        help="random seed")
    # args_other.add_argument('--device', type=str, default='cuda:0', choices=['cuda:%d' % device for device in range(device_count())] + ['cpu'],
    #                     help="device to train/validate/valid with")
    args_other.add_argument('--device', type=str, default='cuda:0', choices=['cuda:%d' % device for device in range(device_count())] + ['cpu'],
                        help="device to train/validate/valid with")
    args_other.add_argument('--resume', action='store_true', default=False,
                        help="resume experiment from save/NAME checkpoint")
    args_other.add_argument('--help', '-h', action='store_true', default=False,
                        help="show this help message and exit")

    args = parser.parse_args()
    if args.help:
        parser.print_help()
        exit()

    if args.iters is not None:
        args.rounds = sys.maxsize

    if args.name is None:
        args.name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{args.dataset}-rounds{args.rounds}"

    # args.dataset_args = args_str_to_dict(args.dataset_args)
    # args.model_args = args_str_to_dict(args.model_args)
    args.optim_args = args_str_to_dict(args.optim_args)
    args.sched_args = args_str_to_dict(args.sched_args)

    return args

def args_str_to_dict(args_str):
    args_dict = {}
    if args_str is not None:
        for arg in args_str.replace(' ', '').split(','):
            keyvalue = arg.split('=')
            args_dict[keyvalue[0]] = literal_eval(keyvalue[1])
    return args_dict
