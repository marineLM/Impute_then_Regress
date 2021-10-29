'''PyTorch MLP'''

import math
import numpy as np
from sklearn.base import RegressorMixin

import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorchtools import EarlyStopping


class Mlp(nn.Module):
    def __init__(self, layer_sizes, init_type):
        super().__init__()
        self.relu = nn.ReLU()

        # Create the parameters for the MLP
        l_W_mlp = []
        for d_in, d_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            l_W_mlp.append(torch.empty(d_in, d_out, dtype=torch.double))

        l_b_mlp = [torch.empty(d, dtype=torch.double) for d in layer_sizes[1:]]

        # Initialize randomly the parameters of the MLP
        if init_type == 'normal':
            for W in l_W_mlp:
                if len(W.shape) >= 2:
                    nn.init.xavier_normal_(W, gain=math.sqrt(2))
                else:
                    tmp = math.sqrt(2/(W.shape[0]+1))
                    nn.init.normal_(W, mean=0, std=tmp)

        elif init_type == 'uniform':
            for W in l_W_mlp:
                if len(W.shape) >= 2:
                    nn.init.xavier_uniform_(W, gain=math.sqrt(2))
                else:
                    tmp = math.sqrt(2*6/(W.shape[0]+1))
                    nn.init.uniform_(W, -tmp, tmp)

        for b_mlp in l_b_mlp:
            nn.init.zeros_(b_mlp)

        # Make tensors learnable parameters
        self.l_W_mlp = [torch.nn.Parameter(W) for W in l_W_mlp]
        for i, W in enumerate(self.l_W_mlp):
            self.register_parameter('W_mlp_{}'.format(i), W)
        self.l_b_mlp = [torch.nn.Parameter(b) for b in l_b_mlp]
        for i, b_mlp in enumerate(self.l_b_mlp):
            self.register_parameter('b_mlp_{}'.format(i), b_mlp)

    def forward(self, x, phase='train'):
        """
        Parameters:
        ----------
        x: tensor, shape (batch_size, (2x)n_features)
            The input data imputed (with the mask concatenated).
        """
        n_layers = len(self.l_W_mlp)

        h = x
        for i in range(n_layers):
            h = torch.matmul(h, self.l_W_mlp[i]) + self.l_b_mlp[i]
            if i < (n_layers - 1):
                h = self.relu(h)

        return h.view(-1)


class MLP_reg(RegressorMixin):
    """

    Parameters
    ----------

    width_factor: int
        The width of the MLP is given by `width_factor` times the number of
        features.

    n_epochs: int
        The maximum number of epochs.

    batch_size: int
        The batch size.

    lr: float
        The learning rate.

    weight_decay: float
        The weight decay parameter.

    early_stopping: boolean
        If True, early stopping is used based on the validaton set.

    optimizer: str
        One of `sgd`or `adam`.

    mlp_depth: int
        The depth of the MLP.

    init_type: str
        The type of initialisation for the parameters. Either 'normal',
        'uniform'.

    verbose: boolean
    """

    def __init__(self, width_factor=1, n_epochs=1000, batch_size=100, lr=0.01,
                 weight_decay=1e-4, early_stopping=False, optimizer='sgd',
                 mlp_depth=0, init_type='normal', is_mask=False,
                 verbose=False):
        self.width_factor = width_factor
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop = early_stopping
        self.optimizer = optimizer
        self.mlp_depth = mlp_depth
        self.init_type = init_type
        self.is_mask = is_mask
        self.verbose = verbose

        self.r2_train = []
        self.mse_train = []
        self.r2_val = []
        self.mse_val = []

    def fit(self, X, y, X_val=None, y_val=None):

        n_samples, n_features_all = X.shape

        if self.is_mask:
            n_features = n_features_all//2
        else:
            n_features = n_features_all

        layer_sizes = tuple(
            [n_features_all] +
            [self.width_factor*n_features]*self.mlp_depth + [1])

        X = torch.as_tensor(X, dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)

        if X_val is not None:
            X_val = torch.as_tensor(X_val, dtype=torch.double)
            y_val = torch.as_tensor(y_val, dtype=torch.double)

        self.net = Mlp(layer_sizes, init_type=self.init_type)

        if self.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                                       weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr,
                                        weight_decay=self.weight_decay)

        self.scheduler = ReduceLROnPlateau(
                            self.optimizer, mode='min', factor=0.2,
                            patience=10, threshold=1e-4)

        if self.early_stop and X_val is not None:
            early_stopping = EarlyStopping(verbose=self.verbose)

        criterion = nn.MSELoss()

        # Train the network
        for i_epoch in range(self.n_epochs):
            if self.verbose:
                print("epoch nb {}".format(i_epoch))

            # Shuffle tensors to have different batches at each epoch
            ind = torch.randperm(n_samples)
            X = X[ind]
            y = y[ind]

            xx = torch.split(X, split_size_or_sections=self.batch_size, dim=0)
            yy = torch.split(y, split_size_or_sections=self.batch_size, dim=0)

            param_group = self.optimizer.param_groups[0]
            lr = param_group['lr']
            if self.verbose:
                print("Current learning rate is: {}".format(lr))
            if lr < 1e-4:
                break

            for bx, by in zip(xx, yy):

                self.optimizer.zero_grad()

                y_hat = self.net(bx)

                loss = criterion(y_hat, by)
                loss.backward()

                # Take gradient step
                self.optimizer.step()

            # Evaluate the train loss
            with torch.no_grad():
                y_hat = self.net(X, phase='test')
                loss = criterion(y_hat, y)
                mse = loss.item()
                self.mse_train.append(mse)

                var = ((y - y.mean())**2).mean()
                r2 = 1 - mse/var
                self.r2_train.append(r2)

                if self.verbose:
                    print("Train loss - r2: {}, mse: {}".format(r2, mse))

            # Evaluate the validation loss
            if X_val is not None:
                with torch.no_grad():
                    y_hat = self.net(X_val, phase='test')
                    loss_val = criterion(y_hat, y_val)
                    mse_val = loss_val.item()
                    self.mse_val.append(mse_val)

                    var = ((y_val - y_val.mean())**2).mean()
                    r2_val = 1 - mse_val/var
                    self.r2_val.append(r2_val)
                    if self.verbose:
                        print("Validation loss is: {}".format(r2_val))

                if self.early_stop:
                    early_stopping(mse_val, self.net)
                    if early_stopping.early_stop:
                        if self.verbose:
                            print("Early stopping")
                        break

                self.scheduler.step(mse_val)

        # load the last checkpoint with the best model
        if self.early_stop and early_stopping.early_stop:
            self.net.load_state_dict(early_stopping.checkpoint)

    def predict(self, X):

        X = torch.as_tensor(X, dtype=torch.double)

        with torch.no_grad():
            y_hat = self.net(X, phase='test')

        return np.array(y_hat)
