'''This code imlpements early stopping and was taken from
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py'''

import numpy as np
from copy import deepcopy


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a
    given patience."""
    def __init__(self, patience=12, verbose=False, delta=1e-4):
        """
        Parameters
        -----------
        patience: int (default: 10)
            How long to wait after last time validation loss improved.

        verbose: bool (default: False)
            If True, prints a message for each validation loss improvement.

        delta: float (default: 1e-4)
            Minimum change in the monitored quantity to qualify as an
            improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        self.checkpoint = deepcopy(model.state_dict())
        self.val_loss_min = val_loss
