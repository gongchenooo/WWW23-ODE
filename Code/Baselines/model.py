from abc import ABC, abstractmethod
import numpy as np
import random

from baseline_constants import ACCURACY_KEY, AVG_LOSS_KEY

from utils.model_utils import batch_data

class Model(ABC):
    def __init__(self, lr, seed, optimizer=None):
        self.lr = lr
        self.optimizer = optimizer
        self.rng = random.Random(seed)
        self.size = None
        self.flops = 0

    def train(self, data, num_epochs=5, batch_size=None, lr=None):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
            avg_loss_list[-1]: average of stochastic loss in the final epoch
        """

        self.optimizer.learning_rate = self.lr if lr is None else lr

        avg_loss_list = []
        for epoch in range(num_epochs):
            tot_loss = []
            for (batched_x, batched_y) in batch_data(data, batch_size, rng=self.rng, shuffle=False):
                input_data = self.process_x(batched_x)
                target_data = self.process_y(batched_y)
                loss = self.optimizer.run_step(input_data, target_data)
                tot_loss.append(loss)
            avg_loss_list.append(sum(tot_loss) / len(tot_loss))
        #print('Loss:{}',format(avg_loss_list))
        self.optimizer.end_local_updates() # store parameters in w after local training
        update = np.copy(self.optimizer.w - self.optimizer.w_on_last_update)
        #self.optimizer.update_w()

        comp = num_epochs * len(batched_y) * batch_size * self.flops
        return comp, update, avg_loss_list[-1]


    def test(self, eval_data, max_batch_size=None):
        """
        Tests the current model on the given data.
        Args:
            eval_data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        if max_batch_size is None:
            max_batch_size = 500
        max_batch_size = 1000
        total_loss, total_correct, count = 0.0, 0, 0
        for (batched_x, batched_y) in batch_data(eval_data, max_batch_size, rng=self.rng, shuffle=False):
            x_vecs = self.process_x(batched_x)
            labels = self.process_y(batched_y)
            loss = self.optimizer.loss(x_vecs, labels)
            correct = self.optimizer.correct(x_vecs, labels)

            total_loss += loss * labels.shape[0]  # loss returns average over batch
            total_correct += correct  # eval_op returns sum over batch
            count += labels.shape[0]
        
        loss = total_loss / count
        acc = total_correct / count
        return {ACCURACY_KEY: acc, AVG_LOSS_KEY: loss}

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return np.asarray(raw_x_batch)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return np.asarray(raw_y_batch)
    
    def loss(self, x, y):
        x, y = self.process_x(x), self.process_y(y)
        return self.optimizer.loss(x, y)

    def gradient(self, x, y):
        x, y = self.process_x(x), self.process_y(y)
        return self.optimizer.gradient(x, y)


class Optimizer(ABC):
    
    def __init__(self, starting_w=None, loss=None, loss_prime=None):
        '''
        self.w to store updated local model
        self.w_on_last_update to store initial local model, i.e. global model
        '''
        self.w = starting_w
        self.w_on_last_update = np.copy(starting_w)
        self.optimizer_model = None

    @abstractmethod
    def loss(self, x, y):
        return None

    @abstractmethod
    def gradient(self, x, y):
        return None

    @abstractmethod
    def run_step(self, batched_x, batched_y): # should run a first order method step and return loss obtained
        return None

    @abstractmethod
    def correct(self, x, y):
        return None

    def end_local_updates(self):
        pass

    def reset_w(self, w):
        self. w = np.copy(w)
        self.w_on_last_update = np.copy(w)

class ServerModel:
    def __init__(self, model):
        """model: server's model (global model)"""
        self.model = model
        self.rng = model.rng

    @property
    def size(self):
        return self.model.optimizer.size()

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients
        Args:
            clients: list of Client objects
        """
        for c in clients:
            c.model.optimizer.reset_w(self.model.optimizer.w)
            c.model.size = self.model.optimizer.size()

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros_like(points[0])

        for w, p in zip(weights, points):
            weighted_updates += w * p
            #weighted_updates += (w / tot_weights) * p
        weighted_updates /= tot_weights
        return weighted_updates
    
    def update(self, updates, aggregation='FedAvg'):
        """Updates server model using given client updates.
        Args:
            updates: list of (weights, update), where num_samples is the
                number of training samples corresponding to the update, and update
                is a list of variable weights
            aggregation: Algorithm used for aggregation. Allowed values are:
                ['FedAvg'], i.e., only support aggregation with weighted mean
        """
        if len(updates) == 0:
            print('No updates obtained. Continuing without update')
            exit()
        def accept_update(u):
            """define the criterion of accepting updates"""
            # norm = np.linalg.norm([np.linalg.norm(x) for x in u[1]])
            norm = np.linalg.norm(u[1])
            return not (np.isinf(norm) or np.isnan(norm))
        all_updates = updates
        updates = [u for u in updates if accept_update(u)]
        if len(updates) < len(all_updates):
            print('Rejected {} individual updates because of NaN or Inf'.format(len(all_updates) - len(updates)))
        if len(updates) == 0:
            print('All individual updates rejected. Continuing without update')
            return False

        if aggregation == 'FedAvg':
            points = [u[1] for u in updates]
            weights = [u[0] for u in updates]
            weighted_updates = self.weighted_average_oracle(points, weights)
        else:
            exit('We only support FedAvg')

        self.model.optimizer.w += np.array(weighted_updates)
        self.model.optimizer.reset_w(self.model.optimizer.w)  # update server model
        return True


                