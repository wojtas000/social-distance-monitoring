import unittest
import homography
import social_distance_monitoring
import distance_calculation
import cv2
import numpy as np

"""
This module implements class for unit tests.
"""


class TestCalc(unittest.TestCase):
    """
    This class implements standard unit tests for most functions used in the project.
    """

    def test_compute_homography(self):
        """
        Compare compute_homography function (from homography module) with cv2.getPerspectiveTransform
        """
        width, height = 640, 480
        pts1 = ((200, 200), (583, 200), (0, height), (width, height))
        pts2 = ((0, 0), (width, 0), (0, height), (width, height))

        matrix1 = homography.compute_homography(pts1, pts2)
        matrix2 = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))

        threshold = 10**-5

        self.assertTrue((np.linalg.norm(matrix1 - matrix2)) < threshold)

    def test_homography_on_point(self):
        """
        Testing homography_on_point function from homography module
        """

        # test when homography matrix is identity matrix

        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        point = np.array([1, 1])
        point2 = np.array([1, 1])
        self.assertEqual(homography.homography_on_point(point, matrix)[0], point2[0])
        self.assertEqual(homography.homography_on_point(point, matrix)[1], point2[1])

        # compare with cv2.perspectiveTransform

        matrix = np.array([[10, 2, -0.005], [0.01, 5, 0.8], [0.2, 0.3, 1]])
        points = np.array([[123, 258], [376, 329], [528, 184], [477, 89]])
        points2 = np.float32(points).reshape(-1, 1, 2)
        transformed_points = []
        transformed_points2 = np.round(cv2.perspectiveTransform(points2, matrix)).astype(int)
        transformed_points3 = []
        for i in range(len(points)):
            transformed_points.append(homography.homography_on_point(points[i, :], matrix))
            transformed_points3.append([transformed_points2[i][0][0], transformed_points2[i][0][1]])
            self.assertEqual(transformed_points[i][0], transformed_points3[i][0])
            self.assertEqual(transformed_points[i][1], transformed_points3[i][1])

    def test_red_and_green_people(self):
        """
        Testing red_end_green_people function from socialDistanceMonitoring module
        """

        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sdm = social_distance_monitoring.SocialDistanceMonitoring(model=None, image_dataset=None,
                                                                  homography_matrix=matrix,
                                                                  real_pixel_distance=60)
        boxes = [[162.59045, 125.97583, 253.04248, 156.52834],
                 [344.7175, 107.06003, 478.60284, 148.01314],
                 [407.38654, 3.767215, 479.21426, 44.87287],
                 [40.401264, 394.4455, 83.86392, 412.38855],
                 [360.1895, 160.59967, 470.29236, 206.76514],
                 [214.4963, 231.95358, 264.35706, 259.7619]]
        red_people, green_people, _ = sdm.red_and_green_people(boxes)
        red_people2 = [[344.7175, 107.06003, 478.60284, 148.01314], [360.1895, 160.59967, 470.29236, 206.76514]]
        green_people2 = [[162.59045, 125.97583, 253.04248, 156.52834],
                         [407.38654, 3.767215, 479.21426, 44.87287],
                         [40.401264, 394.4455, 83.86392, 412.38855],
                         [214.4963, 231.95358, 264.35706, 259.7619]]
        self.assertListEqual(red_people, red_people2)
        self.assertListEqual(green_people, green_people2)

    def test_min_pixel_distance(self):
        """
        Testing min_pixel_distance function from distance_calculation2 module
        """

        obj1, obj2 = [0, 0], [10, 0]
        homography_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        obj_real_dist = 10
        min_real_dist = 1
        min_dist = distance_calculation.min_pixel_distance(obj1, obj2, homography_matrix, obj_real_dist, min_real_dist)
        self.assertEqual(min_dist, 1)

    def test_social_distance(self):
        """
        Testing social_distance function from distance_calculation2 module
        """

        i = 0
        tests = dict()
        tests['test1'] = {'coordinates': [[1, 1], [2, 2], [2, 2.5], [5, 8]], 'min_distance': 1}
        tests['test2'] = {'coordinates': [[0, 0], [3, 0]], 'min_distance': 2}
        tests['test3'] = {'coordinates': [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], 'min_distance': 1.1}

        cases = [2, 0, 5]
        people = {(2, 2), (2, 2.5)}, set(), {(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)}

        for test in tests.values():
            violating_cases, people_violating, neighbors = distance_calculation.social_distance(test['coordinates'],
                                                                                                test['min_distance'])
            self.assertEqual(cases[i], violating_cases)
            self.assertEqual(people[i], people_violating)
            i += 1

        violating_cases, people_violating, neighbors = distance_calculation.social_distance([[1, 1]], 1)
        self.assertEqual(None, people_violating)


if __name__ == '__main__':
    unittest.main()
