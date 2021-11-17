import numpy as np
import cv2
import copy
from typing import Tuple
from utils.GateEnum import GateEnum


class GateLocationVisualisation:
    def __init__(self, image: np.ndarray, code: int, gate_coordinates: np.ndarray, predicted_code: int,
                 predicted_gate_coordinates: np.ndarray = None):
        """
        This class allows for easy visualisation of model predictions

        :param image: image containing gate
        :param correct_code: correct code representing the location of the gate
        :param gate_coordinates: correct coordinates of the gate
        :param predicted_code: predicted code representing the location of the gate
        :param predicted_gate_coordinates: predicted coordinates of the gate
        """

        # Without this line the program is likely to crash when trying to resize the image
        self.image = np.array(image, dtype='uint8')

        self.image_height, self.image_width, _ = self.image.shape

        # Correct coordinates
        self.gate_location = GateEnum(code)
        self.top_left_corner = (gate_coordinates[0], gate_coordinates[1])
        self.bottom_right_corner = (gate_coordinates[2], gate_coordinates[3])
        self.gate_center = self.__get_gate_center(gate_coordinates[0], gate_coordinates[1],
                                                  gate_coordinates[2], gate_coordinates[3])

        # Predicted values by our model
        self.predicted_gate_location = GateEnum(predicted_code)
        if predicted_gate_coordinates is None:
            self.predicted_top_left_corner = None
            self.predicted_bottom_right_corner = None
            self.predicted_gate_center = None
        else:
            self.predicted_top_left_corner = (predicted_gate_coordinates[0], predicted_gate_coordinates[1])
            self.predicted_bottom_right_corner = (predicted_gate_coordinates[2], predicted_gate_coordinates[3])
            self.predicted_gate_center = self.__get_gate_center(predicted_gate_coordinates[0],
                                                                predicted_gate_coordinates[1],
                                                                predicted_gate_coordinates[2],
                                                                predicted_gate_coordinates[3])

    @staticmethod
    def __get_gate_center(top_left_x, top_left_y, bottom_right_x, bottom_right_y) -> Tuple[int, int]:
        """
        Get the coordinates of the center of the gate

        :param top_left_x: top left x coordinate of the gate
        :param top_left_y: top left y coordinate of the gate
        :param bottom_right_x: bottom right x coordinate of the gate
        :param bottom_right_y: bottom right y coordinate of the gate
        :return: Coordinates of the center of the gate
        """

        width_gate = bottom_right_x - top_left_x
        height_gate = top_left_y - bottom_right_y
        x_coordinate = int(top_left_x + width_gate / 2)
        y_coordinate = int(bottom_right_y + height_gate / 2)

        return x_coordinate, y_coordinate

    # This function is needed because otherwise we will have very small images that are barely visible
    def reshape(self, image_width: int, image_height: int) -> None:
        """
        Reshape image and change all parameters so that they fit the new image

        :param image_width: new width of a image
        :param image_height: new height of a image
        """

        proportion_x = image_width / self.image_width
        propotion_y = image_height / self.image_height

        self.image = cv2.resize(self.image, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        self.image_width = image_width
        self.image_height = image_height

        # Reshaping correct parameters
        self.top_left_corner = (int(self.top_left_corner[0] * proportion_x),
                                int(self.top_left_corner[1] * propotion_y))
        self.bottom_right_corner = (int(self.bottom_right_corner[0] * proportion_x),
                                    int(self.bottom_right_corner[1] * propotion_y))
        self.gate_center = (int(self.gate_center[0] * proportion_x),
                            int(self.gate_center[1] * propotion_y))

        # Reshape predicted parameters if possible
        if self.predicted_top_left_corner is not None and self.predicted_bottom_right_corner is not None:
            self.predicted_top_left_corner = (int(self.predicted_top_left_corner[0] * proportion_x),
                                              int(self.predicted_top_left_corner[1] * propotion_y))
            self.predicted_bottom_right_corner = (int(self.predicted_bottom_right_corner[0] * proportion_x),
                                                  int(self.predicted_bottom_right_corner[1] * propotion_y))
            self.predicted_gate_center = (int(self.predicted_gate_center[0] * proportion_x),
                                          int(self.predicted_gate_center[1] * propotion_y))

    # TO DO
    def show_results(self) -> None:
        """
        Show results which include:
        1. On the left side the image with correct gate location and coordinates (if necessary)
        2. On the right side the image with predicted gate location and coordinates (if necessary)
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        # Image with correct location
        image_correct = copy.deepcopy(self.image)
        if self.gate_location == GateEnum['fully_visible']:
            image_correct = cv2.circle(image_correct, self.gate_center, 10, (0, 0, 255), -1)
            image_correct = cv2.rectangle(image_correct, self.top_left_corner, self.bottom_right_corner, (0, 0, 255), 2)

        text = "Correct gate location: " + self.gate_location.name
        textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
        location_text = ((self.image_width - textsize[0]) // 2, self.image_height - 20)

        image_correct = cv2.putText(image_correct, text, location_text, font, font_scale, (255, 255, 255), thickness)

        # Image with predicted location
        image_predicted = copy.deepcopy(self.image)
        if self.predicted_gate_location == GateEnum['fully_visible']:
            image_predicted = cv2.circle(image_predicted, self.predicted_gate_center, 10, (0, 0, 255), -1)
            image_predicted = cv2.rectangle(image_predicted, self.predicted_top_left_corner,
                                            self.predicted_bottom_right_corner, (0, 0, 255), 2)

        text = "Predicted gate location: " + self.predicted_gate_location.name
        textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
        location_text = ((self.image_width - textsize[0]) // 2, self.image_height - 20)

        image_predicted = cv2.putText(image_predicted, text, location_text, font, font_scale, (255, 255, 255),
                                      thickness)

        # Concatenate horizontally
        both_images = np.concatenate((image_correct, image_predicted), axis=1)

        cv2.imshow("Correct vs Predicted", both_images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
