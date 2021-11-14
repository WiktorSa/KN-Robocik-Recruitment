import numpy as np
import torch
import torch.nn as nn
from os import mkdir
from os.path import join, isdir
from dataloader import get_data_loader
from model import SignModel
#from utils.EarlyStopping import EarlyStopping
#from utils.Train import train
#from utils.Test import test


def set_seed(seed: int):
    """
    Set a seed for torch dependencies so that the output is always the same
    :param seed: seed
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_and_save_model(is_classification_task: bool, directory: str, batch_size: int, lr: float, no_epochs: int,
                         save_directory: str, seed: int) -> None:
    """
    Train the model on given images. When training ends save the model weights in a given directory
    :param is_classification_task: Are we performing classification task.
    If True train classification model. Else train regression model
    :param directory: directory where preprocessed data is stored
    :param batch_size: batch size
    :param lr: learning rate for optimizer
    :param no_epochs: number of epochs that the model will be trained on
    :param save_directory: directory where model weights should be saved
    :param seed: seed
    """

    set_seed(seed)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    """
    train_data = np.load(join(directory, 'train_data.npz'))
    val_data = np.load(join(directory, 'val_data.npz'))
    test_data = np.load(join(directory, 'test_data.npz'))

    # We don't need to ask the user for everything. Some values we can deduce from the used data
    _, in_channels, width, height = train_data['X_train'].shape
    _, out_features = train_data['y_train'].shape

    train_dataloader = create_data_loader(train_data['X_train'], train_data['y_train'], batch_size)
    val_dataloader = create_data_loader(val_data['X_val'], val_data['y_val'], batch_size)
    test_dataloader = create_data_loader(test_data['X_test'], test_data['y_test'], batch_size)

    

    
    model = SignModel(in_channels, out_features, width, height).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(save_model)

    # Create directory if it doesn't exist
    if not isdir(save_model):
        mkdir(save_model)

    train(model, train_dataloader, val_dataloader, criterion, optimizer, DEVICE, no_epochs, early_stopping)
    test(model, test_dataloader, criterion, DEVICE)
    
    """
