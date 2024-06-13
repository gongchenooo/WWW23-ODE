import numpy as np
import random
import copy
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, AVG_LOSS_KEY
from baseline_constants import MAX_UPDATE_NORM
from tqdm import tqdm
import torch
from utils.torch_utils import torch_to_numpy
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

class Server:
    def __init__(self, model, dataset=None):
        self.model = model  # global model of the server.
        self.selected_clients = []
        self.updates = []
        self.rng = model.rng  # use random number generator of the model
        self.total_num_comm_rounds = 0
        self.eta = None
        self.dataset = dataset

    def select_clients(self, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = self.rng.sample(possible_clients, num_clients)
        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, round, num_epochs=1, batch_size=None, minibatch=None, clients=None, lr=None, lmbda=None, choosing_method='FIFO'):

        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd
            lr: learning rate to use
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        losses = []
        for c in clients:
            self.model.send_to([c])  # reset client model, store global parameters in client model's w, w_on_last_update, model
            sys_metrics[c.id][BYTES_READ_KEY] += self.model.size
            if lmbda is not None:
                c._model.optimizer.lmbda = lmbda
            if lr is not None:
                c._model.optimizer.learning_rate = lr
            if 'Coor' in choosing_method:
                comp, num_samples, averaged_loss, update = c.train_Coor(round, num_epochs, batch_size, minibatch, lr)
            else:
                comp, num_samples, averaged_loss, update = c.train(round, num_epochs, batch_size, minibatch, lr)
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
            losses.append(averaged_loss)

            self.updates.append((num_samples, update))
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += self.model.size
            sys_metrics[c.id][AVG_LOSS_KEY] = averaged_loss

        avg_loss = np.nan if len(losses) == 0 else \
            np.average(losses, weights=[c.num_train_samples for c in clients])
        return sys_metrics, avg_loss, losses

    def update_model(self, aggregation='FedAvg'):
        is_updated = self.model.update(self.updates, aggregation)
        self.updates = []
        return is_updated

    def calculate_global_grad(self, clients, flag=False):
        # send current model to clients
        self.model.send_to(clients)
        # calculate accurate local gradient
        global_grad = None
        tot_samples = 0
        for c in clients:
            tot_samples += c.num_train_samples
            grad = c.model.gradient(c.train_data['x'], c.train_data['y'])
            if global_grad is None:
                global_grad = [np.array(g.cpu()) * c.num_train_samples for g in grad]
            else:
                for i in range(len(grad)):
                    global_grad[i] += np.array(grad[i].cpu()) * c.num_train_samples
        # calculate global gradient
        self.global_grad = [i / tot_samples for i in global_grad]
        for c in clients:
            c.global_grad = self.global_grad


        # self.rec = []
        # for c in tqdm(clients):
        #     for i in range(len(c.eval_data['x'])):
        #         grad = c.model.gradient(x=[c.eval_data['x'][i]], y=[c.eval_data['y'][i]])
        #         val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
        #         self.rec.append(val)

    def estimate_global_grad(self, round, clients, choosing_method):
        if round == 0:
            self.model.send_to(clients) # send initial models to clients
            self.global_grad = None
            self.tot_samples = 0
            for c in clients:
                self.tot_samples += c.num_train_samples
                if choosing_method == 'Coor+Estimation2+Sim':
                    grad = self.model.model.gradient(c.train_data['x'], c.train_data['y'], layers=2)
                else:
                    grad = self.model.model.gradient(c.train_data['x'], c.train_data['y'])
                c.last_local_grad = [np.array(g.cpu()) for g in grad]
                if self.global_grad is None:
                    self.global_grad = [np.array(g.cpu()) * c.num_train_samples for g in grad]
                else:
                    for i in range(len(self.global_grad)):
                        self.global_grad[i] += np.array(grad[i].cpu()) * c.num_train_samples
            self.global_grad = [g/self.tot_samples for g in self.global_grad]
            for c in clients:
                c.global_grad = self.global_grad
        else:
            # (1) only update part of global gradient estimator based on uploaded local gradient estimators
            if choosing_method in ['Estimation1', 'Coor+Estimation1']:
                for c in self.selected_clients:
                    for i in range(len(self.global_grad)):
                        self.global_grad[i] += (c.delta_local_grad[i] * c.num_train_samples / self.tot_samples)
                for c in self.selected_clients:
                    c.global_grad = self.global_grad
            # (2) aggregate local gradient estimators to global gradient estimator
            elif choosing_method in ['Estimation2', 'Estimation', 'Coor+Estimation2', 'Coor+Estimation2+Sim']:
                thres = 1e10
                if round < thres:
                    clients = self.selected_clients
                self.global_grad = None
                tot_samples = 0
                for c in clients:
                    tot_samples += c.num_train_samples
                    if self.global_grad is None:
                        self.global_grad = [g*c.num_train_samples for g in c.last_local_grad]
                    else:
                        for i in range(len(self.global_grad)):
                            self.global_grad[i] += c.last_local_grad[i] * c.num_train_samples
                self.global_grad = [g/tot_samples for g in self.global_grad]
                for c in self.selected_clients:
                    c.global_grad = self.global_grad
            
            # (3) aggregate local gradient of current model to global gradient estimator
            if choosing_method in ['Estimation3']:
                alpha = 1.0
                tot_samples = 0
                self.global_grad = None
                for c in self.selected_clients:
                    c.model.optimizer.set_params(c.model.optimizer.w_on_last_update)    
                    '''grad1 = [np.array(g.cpu()) for g in self.model.model.gradient(c.data['x'], c.data['y'])]
                    grad2 = c.last_local_grad
                    grad = [(alpha*grad1[i] + (1-alpha)*grad2[i]) for i in range(len(grad1))]'''
                    grad = [np.array(g.cpu()) for g in self.model.model.gradient(c.train_data['x'], c.train_data['y'])]
                    tot_samples += c.num_train_samples
                    if self.global_grad is None:
                        self.global_grad = [g*c.num_train_samples for g in grad]
                    else:
                        for i in range(len(self.global_grad)):
                            self.global_grad[i] += grad[i] * c.num_train_samples
                self.global_grad = [g/tot_samples for g in self.global_grad]
                for c in self.selected_clients:
                    c.global_grad = self.global_grad
            # (4) only update part of global gradient based on uploaded accurate local gradient over current model
            '''for c in self.selected_clients:
                c.model.optimizer.set_params(c.model.optimizer.w_on_last_update) 
                c.old_grad = c.new_grad
                c.new_grad = [np.array(g.cpu()) for g in self.model.model.gradient(c.train_data['x'], c.train_data['y'])]
                for i in range(len(self.global_grad)):
                    self.global_grad[i] += (c.new_grad[i]-c.old_grad[i]) * c.num_train_samples / self.tot_samples
            for c in self.selected_clients:
                c.global_grad = self.global_grad'''

    def test_model(self, clients_to_test=None):
        """Tests self.model on given clients.
        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            train_and_test: If True, also measure metrics on training data
        """
        if clients_to_test is None:
            clients_to_test = self.selected_clients
        metrics = {}
        for client in clients_to_test:
            c_metrics = client.test(self.model.cur_model)
            metrics[client.id] = c_metrics
        avg_metrics = {}
        for k in c_metrics.keys():
            avg_metrics[k] = np.average([metrics[c.id][k] for c in clients_to_test], weights=[c.num_test_samples for c in clients_to_test])
        return metrics, avg_metrics
    
    def get_clients_info(self, clients=None):
        """Returns the ids, hierarchies, num_train_samples and num_test_samples for the given clients.
        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients
        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_train_samples = {c.id: c.num_train_samples for c in clients}
        num_test_samples = {c.id: c.num_test_samples for c in clients}

        return ids, groups, num_train_samples, num_test_samples

    def update_buffer(self, round, clients):
        for c in clients:
            c.update_buffer(round)
        return {c.id: c.num_data_samples for c in clients}#, self.rec
        #return {c.id: c.num_data_samples for c in clients}, {c.id: c.rec for c in clients}

    def Coordinate(self, clients, noise_coor=[0.5, 0.1]):
        '''
        server coordinate different clients to store different labels
        '''                      # synthetic: (200, 10) CIFAR: (25, 5)
        if self.dataset == 'synthetic':
            self.clients_per_label = 200 # 200
            self.labels_per_client = 10  # 10
        elif self.dataset == 'CIFAR':
            self.clients_per_label = 25 # 25    50
            self.labels_per_client = 5  # 5    10
        elif self.dataset == 'HAR':
            self.clients_per_label = 30 #   25    30
            self.labels_per_client = 6  #   5     5
        elif self.dataset == 'HARBOX':
            self.clients_per_label = 120
            self.labels_per_client = 5
        elif self.dataset == 'fashionmnist':
            self.clients_per_label = 25 # 10    25
            self.labels_per_client = 5  # 2     5
        elif self.dataset == 'shakespeare':
            self.clients_per_label = 150 # 30 * 320 / 80
            self.labels_per_client = 30
        elif self.dataset == 'CIFAR100-20':
            self.clients_per_label = 25 # 25 * 20 =
            self.labels_per_client = 10 # 10 * 50
        # receive rough information about data
        info_labels = {i: [] for i in range(self.model.model.num_classes)}
        info_labels_true = {i: [] for i in range(self.model.model.num_classes)}
        size_labels = {i: 0 for i in range(self.model.model.num_classes)}
        for idx in range(len(clients)): # 每一个客户
            noise = float(noise_coor[1]) if (random.random()<noise_coor[0]) else 0.0
            c = clients[idx]
            tmp_labels = np.unique(c.train_data['y'])
            for y in tmp_labels: # 每一个标签数据
                if self.dataset == 'shakespeare':
                    tmp_num = c.train_data['y'].count(y)
                    if tmp_num <= 5:
                        continue
                    info_labels[ALL_LETTERS.find(y)].append((tmp_num, idx)) # (样本数，客户id)
                    size_labels[ALL_LETTERS.find(y)] += tmp_num
                else:
                    tmp_num = np.where(c.train_data['y']==y)[0].shape[0]
                    info_labels_true[y].append((tmp_num, idx))
                    tmp_num = int(tmp_num * max(0, (1+random.gauss(0, noise))))
                    print(np.where(c.train_data['y']==y)[0].shape[0], tmp_num)
                    info_labels[y].append((tmp_num, idx)) # (样本数，客户id)
                    size_labels[y] += tmp_num
        np.save('log/log_{}/information_{}.npy'.format(self.dataset, noise_coor), [info_labels_true, info_labels])
        # exit()
        # sort labels according to #owners in an increasing order
        sorted_labels = sorted(info_labels.items(), key=lambda i:len(i[1])) # 按照客户数排序，从小到大
        print('Sort Labels:', end='\t')
        print({i[0]: len(info_labels[i[0]]) for i in sorted_labels})
        sorted_labels = [i[0] for i in sorted_labels]
        # sort owners of each label according to #samples in a decreasing order
        for y in sorted_labels:
            print('Sort Clients of {}:'.format(y), end='\t')
            info_labels[y] = sorted(info_labels[y], key=lambda j:j[0], reverse=True) # 按照样本数排序，从大到小
            print(info_labels[y][:5])
        # distribute labels to clients
        coordination = {idx: [] for idx in range(len(clients))}
        for y in sorted_labels:
            cnt = 0
            for (num, idx) in info_labels[y]:
                if num == 0:
                    break
                if cnt >= self.clients_per_label: # Distribute to sufficient clients
                    break
                if len(coordination[idx]) == self.labels_per_client: # Be distributed sufficient labels
                    continue
                cnt += 1
                coordination[idx].append(y)
        for k in coordination.keys():
            print('{}:\t{}'.format(k, coordination[k]))
        #print('Calculate new weight for each label')
        # calculate weight for each label
        stored_num = {y: 0 for y in range(self.model.model.num_classes)}
        real_num = size_labels
        for idx in coordination.keys():
            c = clients[idx]
            tmp_size = 0
            # evenly distribute local buffer
            if self.dataset == 'shakespeare':
                coordination[idx] = sorted(coordination[idx], key=lambda y: c.train_data['y'].count(ALL_LETTERS[y]), reverse=True)
            else:
                coordination[idx] = sorted(coordination[idx], key=lambda y: np.where(c.train_data['y']==y)[0].shape[0], reverse=True)
            for y in coordination[idx]: 
                stored_num[y] += int(c.buffer_size / len(coordination[idx]))
                tmp_size += int(c.buffer_size / len(coordination[idx]))
            # distribute remained local buffer
            for y in coordination[idx]:
                if tmp_size == c.buffer_size:
                    break
                else:
                    tmp_size += 1
                    stored_num[y] += 1
        stored_tot = sum(stored_num.values())
        real_tot = sum(real_num.values())
        weight = {}
        for y in range(self.model.model.num_classes):
            if stored_num[y] == 0:
                continue
            
            weight[y] = (real_num[y]/real_tot) / (stored_num[y]/stored_tot) # B_y * (N_y/N) / (B_y/B) = B * (N_y/N)
        print('Label weight:', weight)
        for idx in range(len(clients)):
            c = clients[idx]
            c.distributed_labels = {np.int64(y):weight[y] for y in coordination[idx]}
        return True

    def collect_buffer(self, round, clients=None):
        if clients is None:
            clients = self.selected_clients
        buffer_idx = {}
        for c in clients:
            buffer_idx[c.id] = copy.deepcopy(c.buffer_idx)
        return buffer_idx