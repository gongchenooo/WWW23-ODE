"""Script to run the baselines."""
# python main.py -dataset HARBOX -model log_reg --num-rounds 1000 --eval-every 20 --clients-per-round 6 --num-epochs 5 -lr 0.001 --gpu 0 --choosing-method Coor+FIFO --buffer-size 5
# python main.py -dataset fashionmnist -model LeNet --num-rounds 2000 --eval-every 20 --clients-per-round 5 --num-epochs 2 -lr 0.001 --gpu 0 --choosing-method Coor+FIFO --buffer-size 5
import argparse
import copy
import importlib
import gc
import math
import pickle as pkl
import random
import os
import sys
import time
import json
from datetime import timedelta

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS

from client import Client
from model import ServerModel
from server import Server
from utils.model_utils import setup_clients
from args import parse_args

def main():
    args = parse_args()
    # set seed
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load model
    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # create two different models: one for server and one for each client
    print('Obtained Model Path : ' + str(model_path))
    model_params = MODEL_PARAMS[model_path] # [lr, num_classes]
    model_params_list = list(model_params)
    model_params_list[0] = args.lr
    model_params = tuple(model_params_list)
    print(args)
    client_model = ClientModel(*model_params, seed=args.seed)
    server_model = ServerModel(ClientModel(*model_params, seed=args.seed))
    # set device
    if args.gpu is not None and args.gpu >= 0 and args.gpu <= 4:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(device)
    if hasattr(client_model, 'set_device'):
        client_model.set_device(device)
        server_model.model.set_device(device)

    # record time
    start_time = time.time()

    # create server
    server = Server(server_model, args.dataset)

    # create clients
    clients = setup_clients(args.dataset, model_name=args.model, model=client_model, 
                            num_rounds=args.num_rounds, buffer_size=args.buffer_size, choosing_method=args.choosing_method, 
                            noisy_clients_ratio=args.noisy_clients_ratio, noisy_data_ratio=args.noisy_data_ratio, heterogeneity=args.heterogeneity)

    # Logging
    clients_ids, clients_groups, clients_num_train_samples, clients_num_test_samples = server.get_clients_info(clients)
    if args.noisy_clients_ratio <= 0:
        if args.heterogeneity == 'data':
            log_path = os.path.join('log', 'log_{}'.format(args.dataset), 
                    'clients={}_epoch={}_lr={}'.format(args.clients_per_round, args.num_epochs, args.lr), 
                    args.choosing_method
                    )
        else:
            exit('heterogeneity wrong!')
        noise_coor = [float(i) for i in args.noise_coor.split('-')]
    else:
        print('Experiment with Noisy Data')
        log_path = os.path.join('log_noisy', 
            'log_{}_{}_{}'.format(args.dataset, args.noisy_clients_ratio, args.noisy_data_ratio), 
            'clients={}_epoch={}_lr={}'.format(args.clients_per_round, args.num_epochs, args.lr), 
            args.choosing_method
            )
    save_path = os.path.join('checkpoints', args.dataset, 
        'clients={}_epoch={}_lr={}'.format(args.clients_per_round, args.num_epochs, args.lr), 
        args.choosing_method
        )
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if noise_coor[0] == 0:
        if args.seed == 0:
            file_info = open(log_path + '/info_{}_{}.txt'.format(args.buffer_size, args.batch_size), mode='w')
            file_record = open(log_path + '/record_{}_{}.txt'.format(args.buffer_size, args.batch_size), mode='w')
        else:
            file_info = open(log_path + '/info_{}_{}_{}.txt'.format(args.buffer_size, args.batch_size, args.seed), mode='w')
            file_record = open(log_path + '/record_{}_{}_{}.txt'.format(args.buffer_size, args.batch_size, args.seed), mode='w')
    else:
        if args.seed == 0:
            file_info = open(log_path + '/info_{}_{}_{}.txt'.format(args.buffer_size, args.batch_size, noise_coor), mode='w')
            file_record = open(log_path + '/record_{}_{}_{}.txt'.format(args.buffer_size, args.batch_size, noise_coor), mode='w')
        else:
            file_info = open(log_path + '/info_{}_{}_{}_{}.txt'.format(args.buffer_size, args.batch_size, noise_coor, args.seed), mode='w')
            file_record = open(log_path + '/record_{}_{}_{}_{}.txt'.format(args.buffer_size, args.batch_size, noise_coor, args.seed), mode='w')
    print('Dataset:{}\tModel:{}\tChoosingMethod:{}\tseed:{}\tnoisy_clients_ratio:{}\tnoisy_data_ratio:{}'.format(args.dataset, args.model, args.choosing_method, args.seed, args.noisy_clients_ratio, args.noisy_data_ratio), file=file_info)
    print('Num Rounds:{}\tNum Epochs:{}\tlr_decay:{}\tdecay_lr_every:{}\t'.format(args.num_rounds, args.num_epochs, args.lr_decay, args.decay_lr_every), file=file_info)
    for c in clients:
        print('id:{}\tNum train samples:{}\tNum test samples:{}\tSpeed:{}'.format(c.id, c.num_train_samples, c.num_test_samples, c.speed), file=file_info) 
    file_info.close()
    

    # Coordination
    if args.choosing_method in ['Coor+FIFO', 'Coor+Dream', 'Coor+Estimation1', 'Coor+Estimation2', 'Coor+Estimation2+Sim', 'Coor+FIFO']:
        server.Coordinate(clients, noise_coor=noise_coor)

    # Simulate training
    term_list = []
    for round in range(num_rounds):
        # Test the initial global model on every client and save the initial model
        if round == 0:
            if args.choosing_method in ['Coor+Estimation1', 'Coor+Estimation2', 'Coor+Estimation2+Sim', 'Estimation', 'Estimation1', 'Estimation2', 'Estimation3']:
                server.estimate_global_grad(round, clients, args.choosing_method)
            metrics, avg_metrics = server.test_model(clients_to_test=clients)
            print('Testing:\t', avg_metrics)
            print('Testing:\t', avg_metrics, file=file_record)
            ckpt_path = '{}/{}.ckpt'.format(save_path, round)
            torch.save(server.model, ckpt_path)
            print('Model saved in path: {}'.format(ckpt_path))

        print('--- Round [{}/{}]: Training on {} Clients ---'.format(round + 1, num_rounds, clients_per_round))
        print('--- Round [{}/{}]: Training on {} Clients ---'.format(round + 1, num_rounds, clients_per_round), file=file_record)
        
        # update buffer
        if args.choosing_method in ['Dream', 'Coor+Dream']:
            server.calculate_global_grad(clients)
        num_data_samples = server.update_buffer(round, clients)
        # term_list.append(copy.deepcopy(term))
        # np.save('term_{}.npy'.format(args.dataset), term_list)
        '''for c in clients:
            val_idx_tmp.append(copy.deepcopy(c.buffer))'''

        #start_time = time.time()
        # Select clients to train this round and logging selection
        server.select_clients(online(clients), num_clients = clients_per_round)
        print({c.id: num_data_samples[c.id] for c in server.selected_clients}, file=file_record)

        # Simulate server model training on selected clients' data
        sys_metrics, avg_loss, losses = server.train_model( round=round, num_epochs=args.num_epochs, 
                                                            batch_size=args.batch_size, minibatch=args.minibatch, 
                                                            lr=args.lr, lmbda=None, choosing_method=args.choosing_method)
        #print('Avg training loss:{}'.format(avg_loss))
        #print('Avg training loss:{}'.format(avg_loss), file=file_record)

        # Update global gradient estimator
        if args.choosing_method in ['Coor+Estimation1', 'Coor+Estimation2', 'Coor+Estimation2+Sim', 'Estimation', 'Estimation1', 'Estimation2', 'Estimation3']:
            server.estimate_global_grad(round+1, clients, args.choosing_method)
        # Update server model
        is_updated = server.update_model(aggregation=args.aggregation)

        # Experiments for assumption Record buffer information
        # train_buffer_idx.append(server.collect_buffer(round))
        # tot_buffer_idx.append(server.collect_buffer(round, clients))
        # np.save('train_buffer_idx.npy', train_buffer_idx)
        
        # Test model on all clients
        if (round + 1) % eval_every == 0 or (round + 1) == num_rounds:
            metrics, avg_metrics = server.test_model(clients_to_test=clients)
            end_time = time.time()
            print('Testing:\t', avg_metrics, '\tTime:\t', end_time - start_time)
            print('Testing:\t', avg_metrics, '\tTime:\t', end_time - start_time, file=file_record)
        # Adjust learning rate
        if (round + 1) % args.decay_lr_every == 0:
            args.lr *= args.lr_decay    
            if args.dataset == 'synthetic':
                if args.lr < 3e-5:
                    args.lr = 3e-5
            elif args.dataset == 'HARBOX':
                if args.lr < 1e-5:
                    args.lr = 1e-5
            elif args.dataset == 'fashionmnist':
                if args.lr < 5e-4:
                    args.lr = 5e-4
            elif args.dataset == 'CIFAR':
                if args.lr < 0.01:
                    args.lr = 0.01
        # Save Model
        if (round+1) % 250 == 0:
            ckpt_path = '{}/{}.ckpt'.format(save_path, round)
            torch.save(server.model, ckpt_path)
            print('Model saved in path: {}'.format(ckpt_path))

    # Close models and files
    file_record.close()
    #server_model.close()
    #client_model.close()



def online(clients):
    """We assume all users are always online."""
    return clients

if __name__ == '__main__':
    main()
