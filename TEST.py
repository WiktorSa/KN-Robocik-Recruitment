from dataloader.CreateDataLoader import get_data_loader
import numpy as np

if __name__ == '__main__':
    test_data = np.load('preprocessed_data/test_data.npz')
    test = get_data_loader(test_data['images'], test_data['gate_locations'], test_data['gate_coordinates'], 32, False)
    for x, y in test:
        print(x)
        print(y)
        break
