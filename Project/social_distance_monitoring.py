import numpy as np
import cv2
import matplotlib.pyplot as plt
from homography import homography_on_point
from distance_calculation import social_distance

"""
This module implements class used for social distance monitoring.
"""


class SocialDistanceMonitoring:
    """
    This class is used for creating social distance monitoring objects
    attributes:
    model - model for object classification and people detection
    image_dataset - dataset of images, on which we perform social distance monitoring
    homography_matrix - homography for the given image dataset (transforming camera view to birds-eye view)
    real_pixel_distance - minimal distance between people represented in number of pixels
    """

    def __init__(self, model, image_dataset, homography_matrix, real_pixel_distance):
        self.model = model
        self.image_dataset = image_dataset
        self.homography_matrix = homography_matrix
        self.real_pixel_distance = real_pixel_distance

    @staticmethod
    def boxes_to_centroids(boxes):
        """
        Convert list of bounding boxes of objects to centroids
        :param boxes: <list> list of bounding boxes
        :return: <list> list of centroids
        """
        centroids = []
        for i in range(len(boxes)):
            next_box = (int((boxes[i][1] + boxes[i][3]) / 2), int((boxes[i][0] + boxes[i][2]) / 2))
            centroids.append(next_box)
        return centroids

    def red_and_green_people(self, boxes):
        """
        Function deciding which people do and do not violate the rules of social distancing
        :param boxes: <list> list of coordinates (top left and bottom right corner)
                             of bounding boxes for people in image
        :return: lists coordinates of people violating and not violating social distancing rule and all
        their neighbors which do not keep enough distance.
        """

        # bottom_centers - center bottom of boxes,
        # warped_bottom_centers - center bottom of boxes after applying homography

        original_coordinates_of_people_violating = []
        original_coordinates_of_people_not_violating = []
        n = len(boxes)
        bottom_centers = [0] * n
        warped_bottom_centers = [0] * n

        for i in range(n):
            bottom_centers[i] = np.array([(boxes[i][1]+boxes[i][3])/2, boxes[i][2]])
            warped_bottom_centers[i] = homography_on_point(bottom_centers[i], self.homography_matrix)

        number_of_violation_cases, people_violated, neighbors = social_distance(warped_bottom_centers,
                                                                                min_distance=self.real_pixel_distance)

        # we check which people violate rules and which do not. We append them to the lists we return.

        for i in range(n):
            flag = True
            for person in people_violated:
                if person == tuple(warped_bottom_centers[i]):
                    original_coordinates_of_people_violating.append(boxes[i])
                    flag = False
                    break
            if flag:
                original_coordinates_of_people_not_violating.append(boxes[i])

        return (original_coordinates_of_people_violating, original_coordinates_of_people_not_violating,
                neighbors)

    def modify_image(self, image, boxes):
        """
        Applies colored bounding boxes for people detected in the image. Red box for people violating rules and
        green for others. Moreover function draw red lines connecting people which are in too close neighbourhood
        with each other.

        :param image: image we want to modify
        :param boxes: <list> list of bounding boxes of people in the image
        :return: modified image
        """
        red = (0, 0, 255)
        green = (0, 255, 0)
        red_people, green_people, neighbors = self.red_and_green_people(boxes)

        for person in red_people:
            cv2.rectangle(image, (int(person[1]), int(person[0])),
                          (int(person[3]), int(person[2])), color=red, thickness=2)

        for person in green_people:
            cv2.rectangle(image, (int(person[1]), int(person[0])),
                          (int(person[3]), int(person[2])), color=green, thickness=2)

        for i in range(len(neighbors)):
            for j in neighbors[i]:
                cv2.line(image, (int((boxes[i][1]+boxes[i][3])/2), int((boxes[i][0]+boxes[i][2])/2)),
                         (int((boxes[j][1]+boxes[j][3])/2), int((boxes[j][0]+boxes[j][2])/2)), color=red, thickness=2)

    def bird_eye_view(self, boxes):

        """
        This function returns plot representing bird's eye view of an image.
        :param boxes: <list> list of bounding boxes for detected people
        :return: scatter plot representing bird's-eye view of the given image
        """
        height, width = self.image_dataset[0].shape[:2]
        n = len(boxes)
        bottom_centers = [0] * n
        warped_bottom_centers = [0] * n

        for i in range(n):
            bottom_centers[i] = np.array([(boxes[i][1] + boxes[i][3]) / 2, boxes[i][2]])
            warped_bottom_centers[i] = tuple(homography_on_point(bottom_centers[i], self.homography_matrix))

        _, people_violated, _ = social_distance(warped_bottom_centers, min_distance=self.real_pixel_distance)

        red_people = np.array(list(people_violated))
        green_people = np.array(list(set(warped_bottom_centers).difference(people_violated)))

        if red_people.any():
            red_people[:, 1] = height - red_people[:, 1]
        if green_people.any():
            green_people[:, 1] = height - green_people[:, 1]

        fig, ax = plt.subplots()
        plt.xlim([0, width])
        plt.ylim([0, height])

        if red_people.any():
            for i in range(len(red_people)):
                ax.add_patch(plt.Circle(tuple((red_people[i, :])), self.real_pixel_distance,
                                        edgecolor='red', facecolor=(1, 0.8, 0.8), alpha=0.3))
            ax.scatter(red_people[:, 0], red_people[:, 1], c='red', s=4)

        if green_people.any():
            for i in range(len(green_people)):
                ax.add_patch(plt.Circle(tuple((green_people[i, :])), self.real_pixel_distance,
                                        edgecolor='green', facecolor=(0.83, 1, 0.7), alpha=0.3))
            ax.scatter(green_people[:, 0], green_people[:, 1], c='green', s=4)

        return fig

    def find_monitoring_accuracy(self, data):
        """
        The function finds accuracy of social distance monitoring model. It is an accuracy for correct classification of
        people violating and not violating rule of social distancing.
        :param data: csv file, storing information about people violating and not violating rule of social distancing.
        Each record of the file represents one image from our image dataset. For example, first record corresponds
        with first image from SDM.image_dataset directory, second record with second image etc.
        The dataset should contain columns:
        "Img" - number of image,
        "Violating" - list of centroids representing people violating rule of social distancing
        "Not_violating" - list of centroids representing people not violating the rule.
        :return: accuracy of social distance monitoring model
        """

        i = 0

        accuracy = 0
        number_of_people = 0

        for img in self.image_dataset:

            if i == len(data):
                break

            _, scores, classes = self.model.predict(img)
            boxes = self.model.generate_boxes(img)

            red_people, green_people, neighbors = self.red_and_green_people(boxes)
            red_centroids = set(SocialDistanceMonitoring.boxes_to_centroids(red_people))
            green_centroids = set(SocialDistanceMonitoring.boxes_to_centroids(green_people))

            red_record = set(eval(data['Violating'][i]))
            green_record = set(eval(data['Not_violating'][i]))

            correct = len(red_record.intersection(red_centroids)) + len(green_record.intersection(green_centroids))

            accuracy += correct
            number_of_people += (len(red_people) + len(green_people))

            i += 1

        accuracy = (accuracy / number_of_people) * 100

        return accuracy, number_of_people
