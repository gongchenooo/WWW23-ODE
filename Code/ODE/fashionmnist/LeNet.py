from collections import OrderedDict
import copy
import numpy as np
from model import Model, Optimizer
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from utils.torch_utils import numpy_to_torch, torch_to_numpy

class ClientModel(Model):
    def __init__(self, lr, num_classes, seed=None, optimizer=None):
        self.num_classes = num_classes
        self.device = torch.device('cpu') 
        model = LeNet().to(self.device)
        optimizer = FedOptimizer(model)
        super(ClientModel, self).__init__(lr, seed, optimizer)
        
    def set_device(self, device):
        self.device = device
        self.optimizer.set_device(device)
    
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return torch.from_numpy(
            np.asarray(raw_x_batch, dtype=np.float32).reshape([-1, 1, 28, 28])
        ).to(self.device)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        # return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float32), device=device)
        return torch.LongTensor(raw_y_batch).to(self.device)

class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, 5)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c2', nn.Conv2d(6, 16, 5)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        ]))
        self.fc1 = nn.Linear(256, 84)
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        #print(output.shape)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

class FedOptimizer(Optimizer):
    def __init__(self, model):
        super(FedOptimizer, self).__init__(torch_to_numpy(model.trainable_parameters()))
        self.optimizer_model = None
        self.learning_rate = None
        self.lmbda = None
        self.model = model

    def initialize_w(self):
        self.w = torch_to_numpy(self.model.trainable_parameters())
        self.w_on_last_update = np.copy(self.w)

    def reset_w(self, w):
        """w is provided by server; update self.model to make it consistent with this
           w_on_last_update used to store initial w before local model updates
           set parameters to w, w_on_last_update, model
        """
        self.w = np.copy(w)
        self.w_on_last_update = np.copy(w)
        numpy_to_torch(self.w, self.model) # reset model and stored parameters

    def set_params(self, w):
        numpy_to_torch(w, self.model) # only reset model without resetting stored parameters

    def end_local_updates(self):
        """ self.model is updated by epochs; update self.w to make it consistent with this
            store parameters in w
        """
        self.w = torch_to_numpy(self.model.trainable_parameters())

    #def update_w(self):
    #    self.w_on_last_update = self.w

    def _l2_reg_penalty(self):
        # TODO: note: no l2penalty is applied to the convnet
        loss = sum([torch.norm(p)**2  for p in self.model.trainable_parameters()])
        if self.lmbda is not None:
            return 0.5 * self.lmbda * loss
        else:
            return 0

    def loss(self, x, y):
        """Compute average batch loss on processed batch (x, y)"""
        with torch.no_grad():
            preds = self.model(x)
            loss = cross_entropy(preds, y, reduction='mean') + self._l2_reg_penalty()
        return loss.item()

    def gradient(self, x, y, layers=-1):
        """compute batch gradient on processed batch (x, y)"""
        preds = self.model(x)
        loss = cross_entropy(preds, y, reduction='mean') + self._l2_reg_penalty()
        if layers < 0:
            gradient = torch.autograd.grad(loss, self.model.trainable_parameters())
        else:
            gradient = torch.autograd.grad(loss, self.model.trainable_parameters()[-layers:])
        return gradient


    def loss_and_gradient(self, x, y):
        preds = self.model(x)
        loss = cross_entropy(preds, y, reduction='mean') + self._l2_reg_penalty()
        gradient = torch.autograd.grad(loss, self.model.trainable_parameters())
        return loss, gradient

    def run_step(self, batched_x, batched_y):
        """Run single gradient step on (batched_x, batched_y) and return loss encountered"""
        loss, gradient = self.loss_and_gradient(batched_x, batched_y)
        for p, g in zip(self.model.trainable_parameters(), gradient): # update model
            p.data -= self.learning_rate * g.data
        return loss.item()

    def run_step_Coor(self, batched_x, batched_y, distributed_labels):
        """Run single gradient step on (batched_x, batched_y) and return loss encountered
           each class has unique weight according to coordination strategy 
        """
        tot_gradient = None
        weight_sum = 0
        tot_loss = 0
        for y in distributed_labels.keys():
            idx_list = np.where(batched_y.cpu()==y)[0]
            if len(idx_list) == 0:
                continue
            tmp_batched_x = batched_x[idx_list]
            tmp_batched_y = batched_y[idx_list]
            loss, gradient = self.loss_and_gradient(tmp_batched_x, tmp_batched_y)
            weight = idx_list.shape[0] * distributed_labels[y]
            weight_sum += weight
            tot_loss += loss * idx_list.shape[0]
            if tot_gradient is None:
                tot_gradient = [g.data*weight for g in gradient]
            else:
                for i in range(len(tot_gradient)):
                    tot_gradient[i] += gradient[i].data * weight
        tot_gradient = [g/weight_sum for g in tot_gradient]
        tot_loss /= batched_y.shape[0]
        for p, g in zip(self.model.trainable_parameters(), tot_gradient): # update model
            p.data -= self.learning_rate * g
        return tot_loss.item()

    def correct(self, x, y):
        """Compute current num"""
        with torch.no_grad():
            outputs = self.model(x)
            pred = outputs.argmax(dim=1, keepdim=True)
            return pred.eq(y.view_as(pred)).sum().item()

    def size(self):
        return len(self.w)

    def set_device(self, device):
        self.model = self.model.to(device)