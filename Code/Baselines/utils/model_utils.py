import json
import numpy as np
import os
from collections import defaultdict
import sys
sys.path.append("..")
from client import Client
sys.path.pop()
import random
from tqdm import tqdm
import copy

def batch_data(data, batch_size, rng=None, shuffle=False):
    """
    Args:
        data: {'x': [], 'y': []}
    Return:
        x, y which are both numpy array of length batch_size
    """
    # shuffle data
    x, y = data['x'], data['y']
    x_y = list(zip(x, y))
    if shuffle:
        assert rng is not None
        rng.shuffle(x_y)
    x, y = zip(*x_y)

    if batch_size == -1 or batch_size is None:
        batch_size = len(y)
        
    for i in range(0, len(x), batch_size):
        batched_x = x[i: i+batch_size]
        batched_y = y[i: i+batch_size]
        yield (batched_x, batched_y)

def setup_clients(dataset, model_name, model=None, num_rounds=1000, buffer_size=100, 
                  choosing_method='FIFO', noisy_clients_ratio=0, noisy_data_ratio=0):
    """
    Instantiate clients based on given train and test data directories
    Return:
        all clients: list of Client Objects
    """
    train_data_dir = os.path.join('..', '..', 'Data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', '..', 'Data', dataset, 'data', 'test')
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    clients = create_clients(users, groups, dataset, train_data, test_data, model, num_rounds, choosing_method, buffer_size, noisy_clients_ratio, noisy_data_ratio)
    return clients


def generate_noise(data, noisy_data_ratio):
    length = len(data['y'])
    noisy_idx = np.random.choice(range(length), int(length * noisy_data_ratio))
    for i in noisy_idx:
        data['y'][i] = (data['y'][i] + i) % 10
        # data['y'][i] = (data['y'][i] + i) % 5
    return data

def create_clients(users, groups, dataset, train_data, test_data, model, num_rounds, choosing_method, buffer_size, noisy_clients_ratio=0, noisy_data_ratio=0):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = []
    print('-'*100)
    print('Generate Noisy Data:', end='\t')
    for u, g in tqdm(zip(users, groups)):
        r = np.random.rand()
        if r < noisy_clients_ratio:
            train_data[u] = generate_noise(train_data[u], noisy_data_ratio)
            print(u, end='\t')
        # 每个人具有一个单独的模型
        clients.append(Client(u, g, dataset, train_data[u], test_data[u], copy.deepcopy(model), choosing_method, buffer_size, num_rounds))
        #clients.append(Client(u, g, train_data[u], test_data[u], model, choosing_method, buffer_size, num_rounds))
    print('-'*100)
    return clients

def read_data(train_data_dir, test_data_dir):
    """
    assumes:
        data in input directories are .json files with keys 'users' and 'user_data'
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)
    assert train_clients == test_clients
    assert train_groups == test_groups
    return train_clients, train_groups, train_data, test_data

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)
    files = os.listdir(data_dir) # 里面有多个.json数据文件
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])
    clients = list(sorted(data.keys()))
    return clients, groups, data
