import numpy as np
import torch
from os.path import join


class EarlyStopping:
    def __init__(self, path: str, patience: int = 7):
        """
        Implement a basic early stopping algorithm to improve learning process
        :param path: path where model weights should be saved
        :param patience: how long to wait after last validation loss decrease
        """

        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_val_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss: float, model):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            # Save the weights of the model
            torch.save(model.state_dict(), join(self.path, 'model_weights.pth'))

        elif self.counter >= self.patience:
            print("Val loss didn't improve in {iter} iterations. Early stopping.".format(iter=self.patience))
            # Load the last saved weights (weights that had the best val loss)
            model.load_state_dict(torch.load(join(self.path, 'model_weights.pth')))
            self.early_stop = True

        else:
            self.counter += 1
