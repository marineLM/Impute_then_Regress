'''NeuMiss + MLP'''

import math
import numpy as np
from sklearn.base import BaseEstimator

import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorchtools import EarlyStopping


class Neumiss(nn.Module):
    def __init__(self, n_features, mode, depth, residual_connection, mlp_depth,
                 width_factor, init_type, add_mask, Sigma_gt, mu_gt, beta_gt,
                 beta0_gt, L_gt, tmu_gt, tsigma_gt, coefs):
        super().__init__()
        self.n_features = n_features
        self.mode = mode
        self.depth = depth
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.width_factor = width_factor
        self.relu = nn.ReLU()
        self.add_mask = add_mask

        # NeuMiss block
        # -------------
        # Create the parameters of the network
        if init_type != 'custom_normal':
            if self.mode == 'baseline':
                l_W = [torch.empty(n_features, n_features, dtype=torch.double)
                       for _ in range(self.depth)]
            else:
                W = torch.empty(n_features, n_features, dtype=torch.double)
            Wc = torch.empty(n_features, n_features, dtype=torch.double)
            mu = torch.empty(n_features, dtype=torch.double)
        else:
            if Sigma_gt is None or mu_gt is None or L_gt is None:
                raise ValueError('With custom initialisation, Sigma, mu' +
                                 'and L must be specified.')
            Sigma_gt = torch.as_tensor(Sigma_gt, dtype=torch.double)
            W = torch.eye(n_features, dtype=torch.double) - Sigma_gt*2/L_gt
            Wc = Sigma_gt*2/L_gt
            mu = torch.as_tensor(mu_gt, dtype=torch.double)

        if self.mlp_depth > 0:
            beta = torch.empty(width_factor*n_features, dtype=torch.double)
        elif add_mask:
            beta = torch.empty(2*n_features, dtype=torch.double)
        else:
            beta = torch.empty(n_features, dtype=torch.double)
        b = torch.empty(1, dtype=torch.double)
        coefs = torch.ones(self.depth+1, dtype=torch.double)

        # Initialize randomly the parameters of the network
        if init_type == 'normal':
            if self.mode == 'baseline':
                for W in l_W:
                    nn.init.xavier_normal_(W, gain=0.5)
            else:
                nn.init.xavier_normal_(W, gain=0.5)
            nn.init.xavier_normal_(Wc)

            tmp = math.sqrt(2/(beta.shape[0]+1))
            nn.init.normal_(beta, mean=0, std=tmp)

        elif init_type == 'uniform':
            if self.mode == 'baseline':
                for W in l_W:
                    nn.init.xavier_uniform_(W, gain=0.5)
            else:
                nn.init.xavier_uniform_(W, gain=0.5)
            nn.init.xavier_uniform_(Wc)

            tmp = math.sqrt(2*6/(beta.shape[0]+1))
            nn.init.uniform_(beta, -tmp, tmp)

        elif init_type == 'custom_normal':
            tmp = math.sqrt(2/(beta.shape[0]+1))
            nn.init.normal_(beta, mean=0, std=tmp)

        nn.init.zeros_(mu)
        nn.init.zeros_(b)

        # Make tensors learnable parameters
        if self.mode == 'baseline':
            self.l_W = [torch.nn.Parameter(W) for W in l_W]
            for i, W in enumerate(self.l_W):
                self.register_parameter('W_{}'.format(i), W)
        else:
            self.W = torch.nn.Parameter(W)
        self.Wc = torch.nn.Parameter(Wc)
        self.beta = torch.nn.Parameter(beta)
        self.mu = torch.nn.Parameter(mu)
        self.b = torch.nn.Parameter(b)
        self.coefs = torch.nn.Parameter(coefs)

        if mode != 'shared_accelerated':
            self.coefs.requires_grad = False

        # MLP after the NeuMiss block
        # ---------------------------
        # Create the parameters for the MLP added after the NeuMiss block
        width = width_factor*n_features
        if self.add_mask:
            n_input = 2*n_features
        else:
            n_input = n_features
        l_W_mlp = [torch.empty(n_input, width, dtype=torch.double)]
        for _ in range(mlp_depth - 1):
            l_W_mlp.append(torch.empty(width, width, dtype=torch.double))
        l_b_mlp = [torch.empty(width, dtype=torch.double)
                   for _ in range(mlp_depth)]

        # Initialize randomly the parameters of the MLP
        if init_type in ['normal', 'custom_normal']:
            for W in l_W_mlp:
                nn.init.xavier_normal_(W, gain=math.sqrt(2))

        elif init_type == 'uniform':
            for W in l_W_mlp:
                nn.init.xavier_uniform_(W, gain=math.sqrt(2))

        for b_mlp in l_b_mlp:
            nn.init.zeros_(b_mlp)

        # Make tensors learnable parameters
        self.l_W_mlp = [torch.nn.Parameter(W) for W in l_W_mlp]
        for i, W in enumerate(self.l_W_mlp):
            self.register_parameter('W_mlp_{}'.format(i), W)
        self.l_b_mlp = [torch.nn.Parameter(b) for b in l_b_mlp]
        for i, b in enumerate(self.l_b_mlp):
            self.register_parameter('b_mlp_{}'.format(i), b)

    def forward(self, x, m, phase='train'):
        """
        Parameters:
        ----------
        x: tensor, shape (batch_size, n_features)
            The input data imputed by 0.
        m: tensor, shape (batch_size, n_features)
            The missingness indicator (0 if observed and 1 if missing).
        """
        # h0 = x + m*self.mu
        h = x - (1-m)*self.mu
        h_res = x - (1-m)*self.mu

        h = h*self.coefs[0]

        for i in range(self.depth):
            if self.mode == 'baseline':
                self.W = self.l_W[i]
            h = torch.matmul(h, self.W)*(1-m)
            is_baseline_and_init = (i == 0) and (self.mode == 'baseline')
            if self.residual_connection and not is_baseline_and_init:
                h += h_res*self.coefs[i+1]

        # h = torch.matmul(h, self.Wc)*m + h0
        if self.add_mask:
            h = torch.cat((h, m), 1)

        if self.mlp_depth > 0:
            for W, b in zip(self.l_W_mlp, self.l_b_mlp):
                h = torch.matmul(h, W) + b
                h = self.relu(h)

        y = torch.matmul(h, self.beta)

        y = y + self.b

        return y


class Neumiss_mlp(BaseEstimator):
    """The Neumiss + MLP neural network

    Parameters
    ----------

    mode: str
        One of:
        * 'baseline': The weight matrices for the Neumann iteration are not
        shared.
        * 'shared': The weight matrices for the Neumann iteration are shared.
        * 'shared_accelerated': The weight matrices for the Neumann iteration
        are shared and one corefficient per residual connection can be learned
        for acceleration.

    depth: int
        The depth of the NeuMiss block.

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

    optimizer: srt
        One of `sgd`or `adam`.

    residual_connection: boolean
        If True, the residual connection of the Neumann network are
        implemented.

    mlp_depth: int
        The depth of the MLP stacked on top of the Neumann iterations.

    width_factor: int
        The width of the MLP stacked on top of the NeuMiss layer is calculated
        as width_factor times n_features.

    init_type: str
        The type of initialisation for the parameters. Either 'normal',
        'uniform', or 'custom_normal'. If 'custom_normal', the values provided
        for the parameter `Sigma`, `mu`, `L` (and `coefs` if accelerated) are
        used to initialise the Neumann block.

    add_mask: boolean
        If True, the mask is concatenated to the output of the NeuMiss block.

    verbose: boolean
    """

    def __init__(self, mode, depth, n_epochs=1000, batch_size=100, lr=0.01,
                 weight_decay=1e-4, early_stopping=False, optimizer='sgd',
                 residual_connection=False, mlp_depth=0, width_factor=1,
                 init_type='normal', add_mask=False, Sigma=None, mu=None,
                 beta=None, beta0=None, L=None, tmu=None, tsigma=None,
                 coefs=None, verbose=False):
        self.mode = mode
        self.depth = depth
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop = early_stopping
        self.optimizer = optimizer
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.width_factor = width_factor
        self.init_type = init_type
        self.add_mask = add_mask
        self.Sigma = Sigma
        self.mu = mu
        self.beta = beta
        self.beta0 = beta0
        self.L = L
        self.tmu = tmu
        self.tsigma = tsigma
        self.coefs = coefs
        self.verbose = verbose

        self.r2_train = []
        self.mse_train = []
        self.r2_val = []
        self.mse_val = []

    def fit(self, X, y, X_val=None, y_val=None):

        M = np.isnan(X)
        X = np.nan_to_num(X)

        n_samples, n_features = X.shape

        if X_val is not None:
            M_val = np.isnan(X_val)
            X_val = np.nan_to_num(X_val)

        M = torch.as_tensor(M, dtype=torch.double)
        X = torch.as_tensor(X, dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)

        if X_val is not None:
            M_val = torch.as_tensor(M_val, dtype=torch.double)
            X_val = torch.as_tensor(X_val, dtype=torch.double)
            y_val = torch.as_tensor(y_val, dtype=torch.double)

        self.net = Neumiss(n_features=n_features, mode=self.mode,
                           depth=self.depth,
                           residual_connection=self.residual_connection,
                           mlp_depth=self.mlp_depth,
                           width_factor=self.width_factor,
                           init_type=self.init_type, add_mask=self.add_mask,
                           Sigma_gt=self.Sigma, mu_gt=self.mu,
                           beta_gt=self.beta, beta0_gt=self.beta0, L_gt=self.L,
                           tmu_gt=self.tmu, tsigma_gt=self.tsigma,
                           coefs=self.coefs)

        if len(list(self.net.parameters())) > 0:
            # Create parameter groups
            group_wd = []
            group_no_wd = []
            for name, param in self.net.named_parameters():
                if name in ['mu', 'b']:
                    group_no_wd.append(param)
                else:
                    group_wd.append(param)

            if self.optimizer == 'sgd':
                self.optimizer = optim.SGD(
                    [{'params': group_wd, 'weight_decay': self.weight_decay},
                     {'params': group_no_wd, 'weight_decay': 0}],
                    lr=self.lr)
            elif self.optimizer == 'adam':
                self.optimizer = optim.Adam(
                    [{'params': group_wd, 'weight_decay': self.weight_decay},
                     {'params': group_no_wd, 'weight_decay': 0}],
                    lr=self.lr)

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
            M = M[ind]
            y = y[ind]

            xx = torch.split(X, split_size_or_sections=self.batch_size, dim=0)
            mm = torch.split(M, split_size_or_sections=self.batch_size, dim=0)
            yy = torch.split(y, split_size_or_sections=self.batch_size, dim=0)

            param_group = self.optimizer.param_groups[0]
            lr = param_group['lr']
            if self.verbose:
                print("Current learning rate is: {}".format(lr))
            if lr < 1e-4:
                break

            for bx, bm, by in zip(xx, mm, yy):

                self.optimizer.zero_grad()

                y_hat = self.net(bx, bm)

                loss = criterion(y_hat, by)
                loss.backward()

                # Take gradient step
                self.optimizer.step()

            # Evaluate the train loss
            with torch.no_grad():
                y_hat = self.net(X, M, phase='test')
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
                    y_hat = self.net(X_val, M_val, phase='test')
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

        M = np.isnan(X)
        X = np.nan_to_num(X)

        M = torch.as_tensor(M, dtype=torch.double)
        X = torch.as_tensor(X, dtype=torch.double)

        with torch.no_grad():
            y_hat = self.net(X, M, phase='test')

        return np.array(y_hat)
