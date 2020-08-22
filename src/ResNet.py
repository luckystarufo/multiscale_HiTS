import torch
import numpy as np
import scipy.interpolate
from utils import DataSet


class NNBlock(torch.nn.Module):
    def __init__(self, arch, activation=torch.nn.ReLU()):
        """
        :param arch: architecture of the nn_block
        :param activation: activation function
        """
        super(NNBlock, self).__init__()

        # param
        self.n_layers = len(arch)-1
        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # network arch
        for i in range(self.n_layers):
            self.add_module('Linear_{}'.format(i), torch.nn.Linear(arch[i], arch[i+1]).to(self.device))

    def forward(self, x):
        """
        :param x: input of nn
        :return: output of nn
        """
        for i in range(self.n_layers - 1):
            x = self.activation(self._modules['Linear_{}'.format(i)](x))
        # no nonlinear activations in the last layer
        x = self._modules['Linear_{}'.format(self.n_layers - 1)](x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self, arch, dt, step_size, activation=torch.nn.ReLU()):
        """
        :param arch: a list that provides the architecture
        :param dt: time step unit
        :param step_size: forward step size
        :param activation: activation function in neural network
        """
        super(ResNet, self).__init__()

        # check consistencies
        assert isinstance(arch, list)
        assert arch[0] == arch[-1]

        # param
        self.n_dim = arch[0]

        # data
        self.dt = dt
        self.step_size = step_size

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # layer
        self.activation = activation
        self.add_module('increment', NNBlock(arch, activation=activation))

    def check_data_info(self, dataset):
        """
        :param: dataset: a dataset object
        :return: None
        """
        assert self.n_dim == dataset.n_dim
        assert self.dt == dataset.dt
        assert self.step_size == dataset.step_size

    def forward(self, x_init):
        """
        :param x_init: array of shape batch_size x input_dim
        :return: next step prediction of shape batch_size x input_dim
        """
        return x_init + self._modules['increment'](x_init)

    def uni_scale_forecast(self, x_init, n_steps):
        """
        :param x_init: array of shape n_test x input_dim
        :param n_steps: number of steps forward in terms of dt
        :return: predictions of shape n_test x n_steps x input_dim and the steps
        """
        steps = list()
        preds = list()
        sample_steps = range(n_steps)

        # forward predictions
        x_prev = x_init
        cur_step = self.step_size - 1
        while cur_step < n_steps + self.step_size:
            x_next = self.forward(x_prev)
            steps.append(cur_step)
            preds.append(x_next)
            cur_step += self.step_size
            x_prev = x_next

        # include the initial frame
        steps.insert(0, 0)
        preds.insert(0, torch.tensor(x_init).float().to(self.device))

        # interpolations
        preds = torch.stack(preds, 2).detach().numpy()
        cs = scipy.interpolate.interp1d(steps, preds, kind='linear')
        y_preds = torch.tensor(cs(sample_steps)).transpose(1, 2).float()

        return y_preds

    def train_net(self, dataset, max_epoch, batch_size, w=1.0, lr=1e-3, model_path=None):
        """
        :param dataset: a dataset object
        :param max_epoch: maximum number of epochs
        :param batch_size: batch size
        :param w: l2 error weight
        :param lr: learning rate
        :param model_path: path to save the model
        :return: None
        """
        # check consistency
        self.check_data_info(dataset)

        # training
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        epoch = 0
        best_loss = 1e+5
        while epoch < max_epoch:
            epoch += 1
            # ================= prepare data ==================
            n_samples = dataset.n_train
            new_idxs = torch.randperm(n_samples)
            batch_x = dataset.train_x[new_idxs[:batch_size], :]
            batch_ys = dataset.train_ys[new_idxs[:batch_size], :, :]
            # =============== calculate losses ================
            train_loss = self.calculate_loss(batch_x, batch_ys, w=w)
            val_loss = self.calculate_loss(dataset.val_x, dataset.val_ys, w=w)
            # ================ early stopping =================
            if best_loss <= 1e-8:
                print('--> model has reached an accuracy of 1e-8! Finished training!')
                break
            # =================== backward ====================
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # =================== log =========================
            if epoch % 1000 == 0:
                print('epoch {}, training loss {}, validation loss {}'.format(epoch, train_loss.item(),
                                                                              val_loss.item()))
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    if model_path is not None:
                        print('(--> new model saved @ epoch {})'.format(epoch))
                        torch.save(self, model_path)

        # if to save at the end
        if val_loss.item() < best_loss and model_path is not None:
            print('--> new model saved @ epoch {}'.format(epoch))
            torch.save(self, model_path)

    def calculate_loss(self, x, ys, w=1.0):
        """
        :param x: x batch, array of size batch_size x n_dim
        :param ys: ys batch, array of size batch_size x n_steps x n_dim
        :return: overall loss
        """
        batch_size, n_steps, n_dim = ys.size()
        assert n_dim == self.n_dim

        # forward (recurrence)
        y_preds = torch.zeros(batch_size, n_steps, n_dim).float().to(self.device)
        y_prev = x
        for t in range(n_steps):
            y_next = self.forward(y_prev)
            y_preds[:, t, :] = y_next
            y_prev = y_next

        # compute loss
        criterion = torch.nn.MSELoss(reduction='none')
        loss = w * criterion(y_preds, ys).mean() + (1-w) * criterion(y_preds, ys).max()

        return loss


def multi_scale_forecast(x_init, n_steps, models):
    """
    :param x_init: initial state torch array of shape n_test x n_dim
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :return: a torch array of size n_test x n_steps x n_dim
    
    This function is not used in the paper for low efficiency,
    we suggest to use vectorized_multi_scale_forecast() below.
    """
    # sort models by their step sizes (decreasing order)
    step_sizes = [model.step_size for model in models]
    models = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]

    # parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_extended_steps = n_steps + min(step_sizes)
    sample_steps = range(1, n_steps+1)

    steps = list()
    preds = list()
    steps.insert(0, 0)
    preds.insert(0, torch.tensor(x_init).float().to(device))
    for model in models:
        tmp_steps = list()
        tmp_preds = list()
        for j in range(len(steps)):
            if j < len(steps) - 1:
                end_step = steps[j+1]
            else:
                end_step = n_extended_steps
            # starting point
            cur_step = steps[j]
            cur_x = preds[j]
            tmp_steps.append(cur_step)
            tmp_preds.append(cur_x)
            while True:
                step_size = model.step_size
                cur_step += step_size
                if cur_step >= end_step:
                    break
                cur_x = model(cur_x)
                tmp_steps.append(cur_step)
                tmp_preds.append(cur_x)
        # update new predictions
        steps = tmp_steps
        preds = tmp_preds

    # interpolation
    preds = torch.stack(preds, 2).detach().numpy()
    cs = scipy.interpolate.interp1d(steps, preds, kind='linear')
    y_preds = torch.tensor(cs(sample_steps)).transpose(1, 2).float()

    return y_preds


def vectorized_multi_scale_forecast(x_init, n_steps, models):
    """
    :param x_init: initial state torch array of shape n_test x n_dim
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :return: a torch array of size n_test x n_steps x n_dim,
             a list of indices that are not achieved by interpolations
    """
    # sort models by their step sizes (decreasing order)
    step_sizes = [model.step_size for model in models]
    models = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]

    # we assume models are sorted by their step sizes (decreasing order)
    n_test, n_dim = x_init.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    indices = list()
    extended_n_steps = n_steps + models[0].step_size
    preds = torch.zeros(n_test, extended_n_steps + 1, n_dim).float().to(device)

    # vectorized simulation
    indices.append(0)
    preds[:, 0, :] = x_init
    total_step_sizes = n_steps
    for model in models:
        n_forward = int(total_step_sizes/model.step_size)
        y_prev = preds[:, indices, :].reshape(-1, n_dim)
        indices_lists = [indices]
        for t in range(n_forward):
            y_next = model(y_prev)
            shifted_indices = [x + (t + 1) * model.step_size for x in indices]
            indices_lists.append(shifted_indices)
            preds[:, shifted_indices, :] = y_next.reshape(n_test, -1, n_dim)
            y_prev = y_next
        indices = [val for tup in zip(*indices_lists) for val in tup]
        total_step_sizes = model.step_size - 1

    # simulate the tails
    last_idx = indices[-1]
    y_prev = preds[:, last_idx, :]
    while last_idx < n_steps:
        last_idx += models[-1].step_size
        y_next = models[-1](y_prev)
        preds[:, last_idx, :] = y_next
        indices.append(last_idx)
        y_prev = y_next

    # interpolations
    sample_steps = range(1, n_steps+1)
    valid_preds = preds[:, indices, :].detach().numpy()
    cs = scipy.interpolate.interp1d(indices, valid_preds, kind='linear', axis=1)
    y_preds = torch.tensor(cs(sample_steps)).float()

    return y_preds







