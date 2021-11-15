import numpy as np
import torch
from os.path import join


class EarlyStopping:
    def __init__(self, directory: str, is_classification_task: bool, patience: int = 7):
        """
        Implement a basic early stopping algorithm to improve learning process
        :param directory: directory where model weights should be saved
        :param is_classification_task: are we performing classification task.
        :param patience: how long to wait after last validation loss decrease
        """

        self.patience = patience
        self.directory = directory
        self.counter = 0
        self.best_val_loss = np.Inf

        if is_classification_task:
            self.file_name = "model_weights_classification.pth"

        else:
            self.file_name = "model_weights_regression.pth"

    def __call__(self, val_loss: float, model) -> bool:
        """
        Compare validation loss with the best validation loss. If the validation loss isn't improving
        stop the learning process

        :param val_loss: validation loss
        :param model: the model that is currently trained
        :return: bool telling if the learning should be stopped
        """

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            # Save the weights of the model
            torch.save(model.state_dict(), join(self.directory, self.file_name))
            return False

        elif self.counter >= self.patience:
            print("Val loss didn't improve in {iter} iterations. Early stopping.".format(iter=self.patience))
            # Load the last saved weights (weights that had the best val loss)
            model.load_state_dict(torch.load(join(self.directory, self.file_name)))
            return True

        else:
            self.counter += 1
            return False
