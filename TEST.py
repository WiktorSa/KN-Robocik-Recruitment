from utils import train_and_save_model

if __name__ == '__main__':
    train_and_save_model(True, 'preprocessed_data', 1, 1e-3, 1, 'model_weights', 1001)
