from utils import train_and_save_model
import numpy as np
from model import GateClassificationModel, GateRegressionModel
import torch.nn as nn
from dataloader import get_data_loader

if __name__ == '__main__':
    #train_and_save_model(True, 'preprocessed_data', 1, 1e-3, 1, 'model_weights', 1001)
    train_data = np.load('preprocessed_data/train_data.npz')
    train_dataloader = get_data_loader(train_data['images'], train_data['gate_locations'],
                                       train_data['gate_coordinates'], 2, True)

    model = GateClassificationModel(3, 5)
    criterion = nn.CrossEntropyLoss()

    #model = GateRegressionModel(3, 4)
    #criterion = nn.MSELoss()

    for x, y in train_dataloader:
        output = model.forward(x.float())
        print(output)
        print(x)
        print(y)
        print(y.shape)
        loss = criterion(output, y.long())
        print(loss)
        break
