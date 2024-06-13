import numpy as np
import random

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, AVG_LOSS_KEY
from baseline_constants import MAX_UPDATE_NORM
from tqdm import tqdm

class Server:
    def __init__(self, model):
        self.model = model  # global model of the server.
        self.selected_clients = []
        self.updates = []
        self.rng = model.rng  # use random number generator of the model
        self.total_num_comm_rounds = 0
        self.eta = None

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

    def train_model(self, round, num_epochs=1, batch_size=None, minibatch=None, clients=None, lr=None, lmbda=None):

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
        for c in tqdm(clients):
            c.update_buffer(round)
        return {c.id: c.num_data_samples for c in clients}
    
    def calculate_global_grad(self, round, dataset, clients):
        # send current model to clients
        self.model.send_to(clients)
        # calculate accurate local gradient
        global_grad = None
        tot_samples = 0
        for c in tqdm(clients):
            tot_samples += c.num_train_samples
            grad = c.model.gradient(c.train_data['x'], c.train_data['y'])
            if global_grad is None:
                global_grad = [np.array(g.cpu()) * c.num_train_samples for g in grad]
            else:
                for i in range(len(grad)):
                    global_grad[i] += np.array(grad[i].cpu()) * c.num_train_samples
        # calculate global gradient
        self.global_grad = [i / tot_samples for i in global_grad]
        
        # Experiments  for assumption
        # grad_list = []
        # for c in clients:
        #     for i in range(3000//len(clients)):
        #         grad = c.model.gradient(x=[c.train_data['x'][i]], y=[c.train_data['y'][i]])
        #         grad_list.append([np.array(g.cpu()) for g in grad])
        # np.save('log/log_{}/grad_round={}.npy'.format(dataset, round), [self.global_grad, grad_list])