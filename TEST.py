from preprocessing import preprocess_data
import numpy as np

if __name__ == '__main__':
    preprocess_data('data', 0.7, 0.2, True, 'preprocessed_data', 1001)
