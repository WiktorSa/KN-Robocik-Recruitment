from dataloader.CreateDataLoader import get_data_loader
from model import GateRegressionModel
import numpy as np

if __name__ == '__main__':
    test_data = np.load('preprocessed_data/test_data.npz')
    test = get_data_loader(test_data['images'], test_data['gate_locations'], test_data['gate_coordinates'], 32, False)
    model = GateRegressionModel(3, 4)
    for x, y in test:
        #model.forward(x.float())
        break

    print(sum(p.numel() for p in model.parameters()))
