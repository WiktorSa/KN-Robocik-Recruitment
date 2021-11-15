import numpy as np
import torch
import torch.nn as nn
from os import mkdir
from os.path import join, isdir
from dataloader import get_data_loader
from model import GateClassificationModel, GateRegressionModel
from utils.GateEnum import GateEnum
from utils.EarlyStopping import EarlyStopping
from utils.Train import train_regression, train_classification
from utils.Test import test_regression, test_classification


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

    :param is_classification_task: are we performing classification task.
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

    train_data = np.load(join(directory, 'train_data.npz'))
    val_data = np.load(join(directory, 'val_data.npz'))
    test_data = np.load(join(directory, 'test_data.npz'))

    # We can deduce some info from the data given
    _, in_channels, _, _ = train_data['images'].shape
    out_features_classification = len(GateEnum)
    _, out_features_regression = train_data['gate_coordinates'].shape

    train_dataloader = get_data_loader(train_data['images'], train_data['gate_locations'],
                                       train_data['gate_coordinates'], batch_size, is_classification_task)
    val_dataloader = get_data_loader(val_data['images'], val_data['gate_locations'],
                                     val_data['gate_coordinates'], batch_size, is_classification_task)
    test_dataloader = get_data_loader(test_data['images'], test_data['gate_locations'],
                                      test_data['gate_coordinates'], batch_size, is_classification_task)

    if is_classification_task:
        model = GateClassificationModel(in_channels, out_features_classification)
        criterion = nn.CrossEntropyLoss()

    else:
        model = GateRegressionModel(in_channels, out_features_regression)
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(save_directory, is_classification_task)

    # Create directory if it doesn't exist
    if not isdir(save_directory):
        mkdir(save_directory)

    if is_classification_task:
        train_classification(model, train_dataloader, val_dataloader, criterion, optimizer, DEVICE, no_epochs,
                         early_stopping)
        test_classification(model, test_dataloader, criterion, DEVICE)

    else:
        train_regression(model, train_dataloader, val_dataloader, criterion, optimizer, DEVICE, no_epochs,
                         early_stopping)
        test_regression(model, test_dataloader, criterion, DEVICE)
