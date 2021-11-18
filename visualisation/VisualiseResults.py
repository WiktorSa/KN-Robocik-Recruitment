import matplotlib.pyplot as plt
import numpy as np
import torch
from os.path import join
from model import GateClassificationModel, GateRegressionModel
from utils.GateEnum import GateEnum
from visualisation.GateLocationVisualisation import GateLocationVisualisation


def visualise_results(directory_data: str, directory_models: str, width: int, height: int, seed: int = None) -> None:
    """
    Visualise the performance of both regression and classification models by
    showing the correct gate location and predicted gate location

    :param directory_data: directory with preprocessed test data
    :param directory_models: directory where models' weights are stored
    :param width: what width should the images have when being shown
    :param height: what height should the images have when being shown
    :param seed: Seed. If None than don't use seed
    """

    # Due to the error in torch there will be an error in max_pool2d. This warning should be ignored
    print("They may be errors showing. They should be ignored.")

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
    results = reshape_results(results, width, height)

    # Show images randomly until user decides to stop (seed should be primarily used for testing)
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    rng.shuffle(results)

    no_result = 0
    no_results = len(results)
    show_images = True
    while show_images:
        results[no_result].show_results()

        no_result += 1
        if no_result == no_results:
            no_result = 0

        user_input = input("If you want to show more images press Y or y: ")
        if user_input.lower() != 'y':
            show_images = False

    print("Stopped showing images")


def get_results(images: np.ndarray, gate_locations: np.ndarray, gate_coordinates: np.ndarray,
                classification_model, regression_model) -> np.ndarray:
    """
    Predict the location of the gate using trained models

    :param images: images containing gates (in the format color_channels x height x width)
    :param gate_locations: codes representing the location of the gates
    :param gate_coordinates: the coordinates of the gates
    :param classification_model: model that will predict the gate location (code)
    :param regression_model: model that will predict the gate location (coordinates)
    :return: numpy array containing predicted results in the form of GateLocationVisualisation class
    """

    results = np.empty(shape=len(images), dtype=object)
    # Note - GateLocationVisualisation and models take differently shaped images as an input!
    for i in range(len(images)):
        # Add extra dimension to avoid error caused by not having batch_size dimension
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


def reshape_results(results: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Reshape results so that they could be visible to the user

    :param results: array containing all results (GateLocationVisualisation class)
    :param width: the width of the reshaped result
    :param height: the height of the reshaped result
    :return: reshaped results
    """

    for result in results:
        result.reshape(width, height)

    return results
