import warnings
import numpy as np
import random
import queue
import copy


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
        self.dataset = dataset
        
        # speed of data streams
        if self.dataset == 'synthetic':
            self.speed = self.num_train_samples / 400
        elif self.dataset == 'CIFAR':
            self.speed = self.num_train_samples / 300
        elif self.dataset == 'CIFAR100':
            self.speed = self.num_train_samples / 300
        elif self.dataset == 'CIFAR100-20':
            self.speed = self.num_train_samples / 300
        elif self.dataset == 'fashionmnist':
            self.speed = self.num_train_samples / 500
        elif self.dataset == 'HAR':
            self.speed = self.num_train_samples / 50
        elif self.dataset == 'HARBOX':
            self.speed = self.num_train_samples / 10
        elif self.dataset == 'shakespeare':
            self.speed = self.num_train_samples / 100
        
        # initiate buffer
        self.init_buffer() 
    
    def init_buffer(self):
        # fullfill the initial buffer of each client 
        # with the last buffer_size data samples
        if self.choosing_method == 'FIFO':
            self.buffer = range(self.num_train_samples - self.buffer_size, self.num_train_samples)
        elif self.choosing_method == 'RS':
            self.buffer = list(range(self.num_train_samples - self.buffer_size, self.num_train_samples))
        elif self.choosing_method == 'FullData':
            pass
        elif self.choosing_method in ['HighLoss', 'GradientNorm']:
            self.buffer_idx = []
            self.buffer = queue.PriorityQueue()
            for i in range(self.num_train_samples - self.buffer_size, self.num_train_samples):
                val_idx = [0, i] # initial value is zero
                self.buffer.put(val_idx)
                self.buffer_idx.append(i)
        if self.choosing_method in ['FedBalancer', 'FedBalancer_nostream']:
            # Fullfill the inial buffer
            self.buffer_idx = []
            self.buffer = queue.PriorityQueue()
            self.max_loss = 0
            for i in range(self.num_train_samples - self.buffer_size, self.num_train_samples):
                val = self.model.loss(x=[self.train_data['x'][i]], y=[self.train_data['y'][i]]) # loss
                val_idx = [val, i] # initial value is current loss
                self.buffer.put(val_idx)
                self.buffer_idx.append(i)
                self.max_loss = max(val, self.max_loss) # maintain max_loss to delete noisy samples with high loss
        if self.choosing_method in ["Li", "Li_nostream"]:
            # Fullfill the inial buffer
            self.buffer_idx = []
            self.buffer = queue.PriorityQueue()
            self.model.optimizer.no_grad(2)
            self.val_sum = 0.0
            self.cnt = 0
            # use random 50 samples to obtain the average gradient norm for deleting noisy samples with high norm values
            for i in range(self.num_train_samples - 50, self.num_train_samples):
                g = self.model.gradient(x=[self.train_data['x'][i]], y=[self.train_data['y'][i]]) # gradient
                val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                self.cnt += 1
                self.val_sum += val
            # use random 30 samples to initiate buffer
            for i in range(self.num_train_samples - 30, self.num_train_samples):
                val_idx = [0, i]
                g = self.model.gradient(x=[self.train_data['x'][i]], y=[self.train_data['y'][i]]) # gradient
                val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                if val <= (self.val_sum/self.cnt) * 1.2: # remove noisy samples with high norm
                    self.buffer.put(val_idx)
                    self.buffer_idx.append(i)
                if len(self.buffer_idx) == self.buffer_size:
                    break
            self.model.optimizer.require_grad(2) # value computation only use final layer to save computation cost
        
    def update_buffer(self, round):
        '''=====================================     FIFO     ==========================================='''
        if self.choosing_method == 'FIFO':
            next_idx = int((round+1) * self.speed) % self.num_train_samples# last id of data generated
            if next_idx >= self.buffer_size:
                self.buffer = range(next_idx - self.buffer_size, next_idx)
            else:
                self.buffer = list(range(next_idx)) + list(range(self.num_train_samples - (self.buffer_size-next_idx), self.num_train_samples))
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer], 'y': [self.train_data['y'][i] for i in self.buffer]}
        '''=====================================     RS (similar performance with FIFO)    =========================================='''
        if self.choosing_method == 'RS':
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                if len(self.buffer) < self.buffer_size:
                    self.buffer.append(idx)
                else:
                    # num of data samples generated after last participate round (include original data in buffer)
                    num_tot_data = i - int((self.last_participate_round+1) * self.speed) + self.buffer_size
                    pos = random.randint(0, num_tot_data-1)
                    if pos < self.buffer_size:
                        self.buffer[pos] = idx
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer], 'y': [self.train_data['y'][i] for i in self.buffer]}
        '''=====================================     FullData     ================================================'''
        if self.choosing_method == 'FullData':
            self.data = self.train_data
        '''=====================================     HighLoss     ================================================'''
        if self.choosing_method in ['HighLoss']:
            # set model to previously received global model or local model for data valuation
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update) 
            # update data values if receiving new global model, i.e. participate last round
            if self.last_participate_round == round-1:
                self.buffer2 = queue.PriorityQueue()
                while self.buffer.qsize() != 0:
                    [val, idx] = self.buffer.get() # outdated (value, idx)
                    val = self.model.loss(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # loss
                    self.buffer2.put([val, idx]) # new (value, idx)
                self.buffer = self.buffer2
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                val = self.model.loss(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]])
                new_val_idx = [val, idx]
                old_val_idx = self.buffer.get()
                if old_val_idx[0] <= new_val_idx[0]:
                    self.buffer.put(new_val_idx)
                    self.buffer_idx.remove(old_val_idx[1])
                    self.buffer_idx.append(new_val_idx[1])
                else:
                    self.buffer.put(old_val_idx)
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
        '''=====================================     GradientNorm     ================================================'''
        if self.choosing_method == 'GradientNorm':
            # set model to previous receiving global model or local model for data valuation
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update) 

            # update data values if receiving new global model, i.e. participate last round
            if self.last_participate_round == round-1:
                self.buffer2 = queue.PriorityQueue()
                while self.buffer.qsize() != 0:
                    [val, idx] = self.buffer.get() # obtain stored data
                    g = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                    val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                    self.buffer2.put([val, idx])
                self.buffer = self.buffer2
                del self.buffer2
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                g = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                new_val_idx = [val, idx]
                old_val_idx = self.buffer.get()
                if old_val_idx[0] <= new_val_idx[0]:
                    self.buffer.put(new_val_idx)
                    self.buffer_idx.remove(old_val_idx[1])
                    self.buffer_idx.append(new_val_idx[1])
                else:
                    self.buffer.put(old_val_idx)
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
        '''=====================================     FedBalancer     ================================================'''
        if self.choosing_method == 'FedBalancer':
            # set model to previous receiving global model or local model for data valuation
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update) 
            # update data values if receiving new global model, i.e. participate last round
            if self.last_participate_round == round-1:
                self.buffer2 = queue.PriorityQueue()
                while self.buffer.qsize() != 0:
                    [val, idx] = self.buffer.get() # obtain stored data
                    val = self.model.loss(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # loss
                    self.max_loss = max(val, self.max_loss)
                    if val <= self.max_loss * 0.9 or np.random.rand() <= 0.3: # delete samples with high loss, then save with probability 30%
                        self.buffer2.put([val, idx])
                    else:
                        self.buffer_idx.remove(idx)
                self.buffer = self.buffer2
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                val = self.model.loss(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]])
                self.max_loss = max(val, self.max_loss)
                new_val_idx = [val, idx]
                if val <= self.max_loss*0.9 or np.random.rand() <= 0.3: # delete samples with high loss, then save with probability 30%
                    if self.buffer.qsize() < self.buffer_size:
                        self.buffer.put(new_val_idx)
                        self.buffer_idx.append(idx)
                    else:
                        old_val_idx = self.buffer.get()
                        if old_val_idx[0] <= new_val_idx[0]:
                            self.buffer.put(new_val_idx)
                            self.buffer_idx.remove(old_val_idx[1])
                            self.buffer_idx.append(new_val_idx[1])
                        else:
                            self.buffer.put(old_val_idx)
                elif self.buffer.qsize() <= 2:
                    self.buffer.put(new_val_idx)
                    self.buffer_idx.append(idx)
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
        '''=====================================     FedBalancer_nostream     ================================================'''
        if self.choosing_method == 'FedBalancer_nostream':
            # set model to previous receiving global model or local model for data valuation
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update) 
            # update data values if receiving new global model, i.e. participate last round
            if self.last_participate_round == round-1:
                self.buffer2 = queue.PriorityQueue()
                while self.buffer.qsize() != 0:
                    [val, idx] = self.buffer.get() # obtain stored data
                    val = self.model.loss(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # loss
                    self.max_loss = max(val, self.max_loss)
                    if val <= self.max_loss * 0.9 or np.random.rand() <= 0.3:
                        self.buffer2.put([val, idx])
                    else:
                        self.buffer_idx.remove(idx)
                self.buffer = self.buffer2
            # receive new data
            val_list = []
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                val = self.model.loss(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]])
                val_list.append(val)
            # according to reference paper, we need to delete samples with norm higher than median value
            # thus we record the median value of sample norms
            val_list.sort()
            self.max_loss = val_list[int(len(val_list)//2)] 
            
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                val = self.model.loss(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]])
                new_val_idx = [val, idx]
                if val <= self.max_loss:
                    if self.buffer.qsize() < self.buffer_size:
                        self.buffer.put(new_val_idx)
                        self.buffer_idx.append(idx)
                    else:
                        old_val_idx = self.buffer.get()
                        if old_val_idx[0] <= new_val_idx[0]:
                            self.buffer.put(new_val_idx)
                            self.buffer_idx.remove(old_val_idx[1])
                            self.buffer_idx.append(new_val_idx[1])
                        else:
                            self.buffer.put(old_val_idx)
                elif self.buffer.qsize() <= 2: # if too few samples, save all coming data
                    self.buffer.put(new_val_idx)
                    self.buffer_idx.append(idx)
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
        '''=====================================     Li     ================================================'''
        if self.choosing_method == 'Li':
            # set model to previous receiving global model or local model for data valuation
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update) 
            self.model.optimizer.no_grad(2)
            # update data values if receiving new global model, i.e. participate last round
            if self.last_participate_round == round-1:
                self.buffer2 = queue.PriorityQueue()
                while self.buffer.qsize() != 0:
                    [val, idx] = self.buffer.get() # obtain stored data
                    g = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                    val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                    self.cnt += 1
                    self.val_sum += val
                    if val <= self.val_sum/self.cnt*1.2 or np.random.rand() <= 0.2:
                        self.buffer2.put([val, idx])
                    else:
                        self.buffer_idx.remove(idx)
                self.buffer = self.buffer2
               
            # receive new data
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                g = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                self.cnt += 1
                self.val_sum += val
                new_val_idx = [val, idx]

                if val <= self.val_sum/self.cnt*1.2 or np.random.rand() <= 0.2:
                    if self.buffer.qsize() < self.buffer_size:
                        self.buffer.put(new_val_idx)
                        self.buffer_idx.append(new_val_idx[1])
                    else:
                        old_val_idx = self.buffer.get()
                        if old_val_idx[0] <= new_val_idx[0]:
                            self.buffer.put(new_val_idx)
                            self.buffer_idx.remove(old_val_idx[1])
                            self.buffer_idx.append(new_val_idx[1])
                        else:
                            self.buffer.put(old_val_idx)
                elif self.buffer.qsize() <= 1:
                    self.buffer.put(new_val_idx)
                    self.buffer_idx.append(new_val_idx[1])

            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
            self.model.optimizer.require_grad(2)

        '''=====================================     Li_nostream     ================================================'''
        if self.choosing_method == 'Li_nostream':
            # set model to previous receiving global model or local model for data valuation
            self.model.optimizer.set_params(self.model.optimizer.w_on_last_update) 
            self.model.optimizer.no_grad(2)
            # update data values if receiving new global model, i.e. participate last round
            if self.last_participate_round == round-1:
                self.buffer2 = queue.PriorityQueue()
                while self.buffer.qsize() != 0:
                    [val, idx] = self.buffer.get() # obtain stored data
                    g = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                    val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                    self.cnt += 1
                    self.val_sum += val
                    if val <= self.val_sum/self.cnt*1.2 or np.random.rand() <= 0.2:
                        self.buffer2.put([val, idx])
                    else:
                        self.buffer_idx.remove(idx)
                self.buffer = self.buffer2
               
            # receive new data
            val_list = []
            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                g = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                val_list.append(val)
            val_list.sort()
            self.max_val = val_list[len(val_list)//2]

            for i in range(int(round * self.speed), int((round+1) * self.speed)):
                idx = i % self.num_train_samples
                g = self.model.gradient(x=[self.train_data['x'][idx]], y=[self.train_data['y'][idx]]) # gradient
                val = np.linalg.norm([np.linalg.norm(i.cpu()) for i in g])
                new_val_idx = [val, idx]

                if val <= self.max_val:
                    if self.buffer.qsize() < self.buffer_size:
                        self.buffer.put(new_val_idx)
                        self.buffer_idx.append(new_val_idx[1])
                    else:
                        old_val_idx = self.buffer.get()
                        if old_val_idx[0] <= new_val_idx[0]:
                            self.buffer.put(new_val_idx)
                            self.buffer_idx.remove(old_val_idx[1])
                            self.buffer_idx.append(new_val_idx[1])
                        else:
                            self.buffer.put(old_val_idx)
                elif self.buffer.qsize() <= 1:
                    self.buffer.put(new_val_idx)
                    self.buffer_idx.append(new_val_idx[1])

                    
                # elif self.buffer.qsize() < self.buffer_size:
                #     self.buffer.put(new_val_idx)
                #     self.buffer_idx.append(new_val_idx[1])
            self.data = {'x':[self.train_data['x'][i] for i in self.buffer_idx], 'y': [self.train_data['y'][i] for i in self.buffer_idx]}
            self.model.optimizer.require_grad(2)

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
        num_train_samples = len(self.train_data['y'])
        self.last_participate_round = round
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