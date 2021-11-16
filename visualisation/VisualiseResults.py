import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from os.path import join
from model import GateClassificationModel, GateRegressionModel
from utils.GateEnum import GateEnum
from visualisation.GateLocationVisualisation import GateLocationVisualisation


def visualise_results(directory_data: str, directory_models: str, seed: int = None) -> None:
    """
    Visualise the performance of both regression and classification models by
    showing the correct gate location and predicted gate location

    :param directory_data: directory with preprocessed test data
    :param directory_models: directory where models' weights are stored
    :param seed: Seed. If None than don't use seed
    """

    # Due to the error in torch there will be an error in max_pool2d. This warning should be ignored
    print("They may be errors showing. They should be ignored.")

    # Set seed - this should be primarily used for testing
    if seed is not None:
        random.seed(seed)

    data = np.load(join(directory_data, 'test_data.npz'))

    images = data['images']
    gate_locations = data['gate_locations']
    gate_coordinates = data['gate_coordinates']

    _, in_channels, _, _ = images.shape
    out_features_classification = len(GateEnum)
    _, out_features_regression = gate_coordinates.shape

    classification_model = GateClassificationModel(in_channels, out_features_classification)
    regression_model = GateRegressionModel(in_channels, out_features_regression)

    # We don't need gpu here
    classification_model.load_state_dict(torch.load(join(directory_models, 'model_weights_classification.pth'),
                                                    map_location='cpu'))
    regression_model.load_state_dict(torch.load(join(directory_models, 'model_weights_regression.pth'),
                                                map_location='cpu'))


    results = get_results(images, gate_locations, gate_coordinates, classification_model, regression_model)
    #results = reshape_results(results)


    plt.imshow(results[0].image)
    plt.show()


def get_results(images: np.ndarray, gate_locations: np.ndarray, gate_coordinates: np.ndarray,
                classification_model, regression_model) -> np.ndarray:
    """
    Predict the location of the gate using trained models

    :param images: images containing gates (in the format color_channels x height x width)
    :param gate_locations: codes representing the location of the gate
    :param gate_coordinates: the coordinates of the gate
    :param classification_model: model that will predict the gate location (code)
    :param regression_model: model that will predict gate coordinates
    :return: numpy array containing predicted results in the form of GateLocationVisualisation class
    """

    results = np.empty(shape=len(images), dtype=object)
    # Note - GateLocationVisualisation and models take differently shaped images as an input!
    for i in range(len(images)):
        # We need to add extra dimension to avoid error caused by not having batch_size dimension
        model_input = torch.tensor(images[i], dtype=torch.float)
        model_input = torch.unsqueeze(model_input, 0)

        # Predict gate location (code)
        with torch.no_grad():
            prediction = classification_model(model_input)
            _, predicted_code = torch.max(prediction, 1)
            predicted_code = predicted_code.item()

        # Only predict gate coordinates if we the gate is fully visible
        if predicted_code == GateEnum['fully_visible'].value:
            with torch.no_grad():
                predicted_gate_coordinates = regression_model(model_input)
                predicted_gate_coordinates = predicted_gate_coordinates.squeeze().tolist()
                predicted_gate_coordinates = list(map(int, predicted_gate_coordinates))
                predicted_gate_coordinates = np.array(predicted_gate_coordinates)

            results[i] = GateLocationVisualisation(images[i].transpose(1, 2, 0), gate_locations[i],
                                                   gate_coordinates[i], predicted_code, predicted_gate_coordinates)

        else:
            results[i] = GateLocationVisualisation(images[i].transpose(1, 2, 0), gate_locations[i],
                                                   gate_coordinates[i], predicted_code)

    return results


def reshape_results(results: np.ndarray) -> np.ndarray:
    """
    Reshape results so that they could be visible to the user

    :param results: array containing all results
    :return: reshaped results
    """

    WIDTH = 800
    HEIGHT = 500

    for result in results:
        result.reshape(WIDTH, HEIGHT)

    return results


