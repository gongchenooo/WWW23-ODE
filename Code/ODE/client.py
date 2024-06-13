import warnings
import numpy as np
import random
import queue
import copy
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
class Client:
    def __init__(self, client_id, group=None, dataset=None, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []}, \
        model=None, choosing_method='FIFO', buffer_size=20, num_rounds=1000):
        self._model = model
        self.id = client_id  # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.choosing_method = choosing_method
        self.buffer_size = buffer_size
        self.num_rounds = num_rounds
        self.rng = model.rng  # use random number generator of the model
        self.last_participate_round = -1
        self.distributed_labels = None
        self.new_num_train_samples = self.num_train_samples
        self.dataset = dataset

        
        if self.choosing_method in ['Estimation', 'Coor+Estimation1', 'Coor+Estimation2', 'Coor+Estimation2+Sim', 'Estimation1', 'Estimation2', 'Estimation3', 'Coor+FIFO']:
            self.local_grad_sum = 0
            self.local_grad_cnt = 0
            self.local_grad = 0.0 # local gradient estimator
            self.last_local_grad = 0 # last local gradient estimator

        # set data stream speed
        if self.dataset == 'synthetic':
            self.speed = self.num_train_samples / 400
        elif self.dataset == 'CIFAR':
            self.speed = self.num_train_samples / 300
        elif self.dataset == 'CIFAR100':
            self.speed = self.num_train_samples / 300
        elif self.dataset == 'CIFAR100-20':
            self.speed = self.num_train_samples / 300
        elif self.dataset == 'fashionmnist':
            # self.speed = self.num_train_samples / 80
            self.speed = self.num_train_samples / 500
        elif self.dataset == 'HAR':
            self.speed = self.num_train_samples / 50
        elif self.dataset == 'HARBOX':
            self.speed = self.num_train_samples / 10
        elif self.dataset == 'shakespeare':
            self.speed = self.num_train_samples / 100
        '''print(self._model.optimizer.w[0])
        self._model.optimizer.w[0] += 1'''
    
    def init_buffer(self):
        if self.choosing_method in ['Dream', 'Estimation', 'Estimation1', 'Estimation2', 'Estimation3']:
            # Fullfill the inial buffer
            self.buffer_idx = []
            self.buffer = queue.PriorityQueue()
            for i in range(self.num_train_samples - self.buffer_size, self.num_train_samples):
                val_idx = (-1e4, i)
                self.buffer.put(val_idx)
                self.buffer_idx.append(i)
        if self.choosing_method in ['Coor+Dream', 'Coor+Estimation1', 'Coor+Estimation2', 'Coor+Estimation2+Sim', 'Coor+FIFO']:
            # distribute buffer to labels
            self.buffer_size2 = {y: int(self.buffer_size / len(self.distributed_labels.keys())) for y in self.distributed_labels.keys()} 
            for y in self.distributed_labels.keys():
                if sum(self.buffer_size2.values()) == self.buffer_size:
                    break
                else:
                    self.buffer_size2[y] += 1
            self.buffer_size = self.buffer_size2
            # initiate buffer
            self.buffer = {}
            self.buffer_idx = []
            self.new_num_train_samples = 0.0
            for y in self.distributed_labels.keys(): # initiate buffer for each allocated label
                self.new_num_train_samples += self.distributed_labels[y] * self.buffer_size[y]# new weight for model aggregation
                self.buffer[y] = queue.PriorityQueue(self.buffer_size[y])
                
                if self.dataset == 'shakespeare':
                    i_list = np.where(self.train_data['y']==ALL_LETTERS[y])[0]
                else:
                    i_list = np.where(self.train_data['y']==y)[0]
                for i in i_list[-self.buffer_size[y]:]:
                    val_idx = (-1e4, i)
                    self.buffer[y].put(val_idx)
                    self.buffer_idx.append(i)

    def update_buffer(self, round):
        if round == 0:
            self.init_buffer() # initiate buffer
        '''=====================================     Dream     ================================================'''
        if self.choosing_method == 'Dream':
            rec = []
            # set model to current global model
            self.model.optimizer.set_params(self.model.optimizer.w) 
            # update values of data in buffer
            self.buffer2 = queue.PriorityQueue()
            while self.buffer.qsize() != 0:
                (val, idx) = self.buffer.get() # obtain stored data
                grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                # val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(-4, 0)])
                val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                #val = sum([np.mean(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                self.buffer2.put((val, idx))
            self.buffer = self.buffer2
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                # val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(-4, 0)])
                val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                # val = sum([np.mean(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                new_val_idx = (val, idx)
                old_val_idx = self.buffer.get()
                # term1 = sum([np.sum(np.array(grad[i].cpu())**2) for i in range(len(grad))])
                # term2 = 2 * val
                # rec.append((term1, term2))
                # val1 = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(-2, 0)])
                # val2 = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(-4, 0)])
                # rec.append([val, val2, val1])
                if old_val_idx[0] <= new_val_idx[0]:
                    self.buffer.put(new_val_idx)
                    self.buffer_idx.remove(old_val_idx[1])
                    self.buffer_idx.append(new_val_idx[1])
                else:
                    self.buffer.put(old_val_idx)
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
            self.rec = rec
        '''=====================================     Estimation     ================================================'''
        if self.choosing_method in ['Estimation', 'Estimation1', 'Estimation2', 'Estimation3']:
            # set model to last received global model (w_on_last_update) instead of last locally updated model (w)
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update) 
            # update values of data in buffer if participate in last round (receive a new global model)
            if self.last_participate_round == round-1:
                self.buffer2 = queue.PriorityQueue()
                while self.buffer.qsize() != 0:
                    (val, idx) = self.buffer.get() # obtain stored data
                    grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                    val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                    self.buffer2.put((val, idx))
                    '''# update local gradient estimator
                    if self.local_grad_cnt == 0:
                        self.local_grad_sum = [np.array(g.cpu()) for g in grad]
                        self.local_grad_cnt = 1
                    else:
                        for i in range(len(grad)):
                            self.local_grad_sum[i] += np.array(grad[i].cpu())
                        self.local_grad_cnt += 1'''
                self.buffer = self.buffer2
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                new_val_idx = (val, idx)
                old_val_idx = self.buffer.get()
                if old_val_idx[0] <= new_val_idx[0]:
                    self.buffer.put(new_val_idx)
                    self.buffer_idx.remove(old_val_idx[1])
                    self.buffer_idx.append(new_val_idx[1])
                else:
                    self.buffer.put(old_val_idx)
                # update local gradient estimator
                if self.local_grad_cnt == 0:
                    self.local_grad_sum = [np.array(g.cpu()) for g in grad]
                    self.local_grad_cnt = 1
                else:
                    for i in range(len(grad)):
                        self.local_grad_sum[i] += np.array(grad[i].cpu())
                    self.local_grad_cnt += 1
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}

        '''=====================================   Coor+Dream   ================================================'''
        if self.choosing_method == 'Coor+Dream':
            # set moddel to current global model
            self.model.optimizer.set_params(self.model.optimizer.w) 
            # update values of data in buffer
            for y in self.distributed_labels.keys():
                buffer2 = queue.PriorityQueue()
                while self.buffer[y].qsize() != 0:
                    (val, idx) = self.buffer[y].get() # obtain stored data
                    grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                    # val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(-4, 0)])
                    val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                    # val = sum([np.mean(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                    buffer2.put((val, idx))
                self.buffer[y] = buffer2
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                y = self.train_data['y'][idx]
                if self.dataset == 'shakespeare':
                    y = ALL_LETTERS.find(y)
                if y not in self.distributed_labels.keys():
                    continue
                grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                # val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(-4, 0)])
                val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                # val = sum([np.mean(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                new_val_idx = (val, idx)
                old_val_idx = self.buffer[y].get()
                if old_val_idx[0] <= new_val_idx[0]:
                    self.buffer[y].put(new_val_idx)
                    self.buffer_idx.remove(old_val_idx[1])
                    self.buffer_idx.append(new_val_idx[1])
                else:
                    self.buffer[y].put(old_val_idx)
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}

        '''=====================================   Coor+Estimation   ================================================'''
        if self.choosing_method in ['Coor+Estimation1', 'Coor+Estimation2', 'Coor+Estimation2+Sim']:
            # set model to last received global model
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update)
            # update values of data in buffer
            if self.last_participate_round == round-1:
                for y in self.distributed_labels.keys():
                    buffer2 = queue.PriorityQueue()
                    while self.buffer[y].qsize() != 0:
                        (val, idx) = self.buffer[y].get() # obtain stored data
                        if self.choosing_method == 'Coor+Estimation2+Sim':
                            grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]], layers=2) # gradient
                        else:
                            grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                        val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                        buffer2.put((val, idx))
                        # update local gradient estimator
                        '''if self.local_grad_cnt == 0:
                            self.local_grad_sum = [np.array(g.cpu()) for g in grad]
                            self.local_grad_cnt += 1
                        else:
                            for i in range(len(grad)):
                                self.local_grad_sum[i] += np.array(grad[i].cpu())
                            self.local_grad_cnt += 1'''
                    self.buffer[y] = buffer2
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                y = self.train_data['y'][idx]
                if self.choosing_method == 'Coor+Estimation2+Sim':
                    grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]], layers=2) # gradient
                else:
                    grad = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                # update local gradient estimator
                if self.local_grad_cnt == 0:
                    self.local_grad_sum = [np.array(g.cpu()) for g in grad]
                    self.local_grad_cnt += 1
                else:
                    for i in range(len(grad)):
                        self.local_grad_sum[i] += np.array(grad[i].cpu())
                    self.local_grad_cnt += 1
                # exclude samples without allocated labels
                if y not in self.distributed_labels.keys():
                    # print(y)
                    #exit('Samples without allocated labels!')
                    continue
                val = sum([np.sum(np.array(grad[i].cpu()) * self.global_grad[i]) for i in range(len(grad))])
                new_val_idx = (val, idx)
                old_val_idx = self.buffer[y].get()
                if old_val_idx[0] <= new_val_idx[0]:
                    self.buffer[y].put(new_val_idx)
                    self.buffer_idx.remove(old_val_idx[1])
                    self.buffer_idx.append(new_val_idx[1])
                else:
                    self.buffer[y].put(old_val_idx)
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
        '''=====================================   Coor+FIFO   ================================================'''
        if self.choosing_method in ['Coor+FIFO']:
            # set model to last received global model
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update)
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                y = self.train_data['y'][idx]
                # exclude samples without allocated labels
                if y not in self.distributed_labels.keys():
                    continue
                val = int(i)
                new_val_idx = (val, idx)
                old_val_idx = self.buffer[y].get()
                if old_val_idx[0] <= new_val_idx[0]:
                    self.buffer[y].put(new_val_idx)
                    self.buffer_idx.remove(old_val_idx[1])
                    self.buffer_idx.append(new_val_idx[1])
                else:
                    self.buffer[y].put(old_val_idx)
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
        return self.buffer_idx


    def train(self, round, num_epochs=1, batch_size=None, minibatch=None, lr=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            averaged_loss: loss averaged over each stochastic update of the last epoch
            minibatch: whether to use minibatch
            update: set of weights
        """
        if batch_size is None or batch_size == -1:
            batch_size = self.num_train_samples
        if minibatch is None:
            data = self.data
            comp, update, averaged_loss = self.model.train(data, num_epochs, batch_size, lr)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac * len(self.train_data['x'])))
            # TODO: fix smapling from lists
            xs, ys = zip(*self.rng.sample(list(zip(self.train_data['x'], self.train_data['y'])), num_data))
            data = {'x': xs, 'y': ys}
            comp, update, averaged_loss = self.model.train(data, num_epochs, num_data, lr)
        
        #num_train_samples = len(data['y'])
        self.last_participate_round = round
        if self.choosing_method in ['Dream', 'Estimation', 'Estimation1', 'Estimation2', 'Estimation3']:
            num_train_samples = len(self.train_data['y'])
        if self.choosing_method in ['Estimation', 'Estimation1', 'Estimation2', 'Estimation3']:
            self.local_grad = [g/self.local_grad_cnt for g in self.local_grad_sum] # local grad estimator
            self.delta_local_grad = [self.local_grad[i] - self.last_local_grad[i] for i in range(len(self.local_grad))] # update of estimator
            self.last_local_grad = self.local_grad
            self.local_grad, self.local_grad_sum, self.local_grad_cnt = 0, 0, 0
        
        return comp, num_train_samples, averaged_loss, update

    def train_Coor(self, round, num_epochs=1, batch_size=None, minibatch=None, lr=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            averaged_loss: loss averaged over each stochastic update of the last epoch
            minibatch: whether to use minibatch
            update: set of weights
        """
        if batch_size is None or batch_size == -1:
            batch_size = self.num_train_samples
        if minibatch is None:
            data = self.data
            comp, update, averaged_loss = self.model.train_Coor(data, num_epochs, batch_size, lr, self.distributed_labels)

        self.last_participate_round = round
        if self.choosing_method in ['Coor+Estimation1', 'Coor+Estimation2', 'Coor+Estimation2+Sim']:
            self.local_grad = [g/self.local_grad_cnt for g in self.local_grad_sum]
            self.delta_local_grad = [self.local_grad[i] - self.last_local_grad[i] for i in range(len(self.local_grad))]
            self.last_local_grad = self.local_grad
            self.local_grad, self.local_grad_sum, self.local_grad_cnt = 0, 0, 0

        if self.choosing_method in ['Coor+Dream', 'Coor+Estimation1', 'Coor+Estimation2', 'Coor+Estimation2+Sim', 'Coor+FIFO']:
            num_train_samples = self.new_num_train_samples
        return comp, num_train_samples, averaged_loss, update

    def reinit_model(self):
        self._model.optimizer.initialize_w()

    def test(self, model):
        """Test given model on self.eval_data.
        Args:
            model: model to measure metrics on
        Return:
            dict of metrics returned by the model.
        """
        return model.test(self.eval_data)

    @property
    def num_train_samples(self):
        return len(self.train_data['y'])

    @property
    def num_test_samples(self):
        return len(self.eval_data['y'])
    @property
    def num_data_samples(self):
        return len(self.data['y'])

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model