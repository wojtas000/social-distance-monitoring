import homography
import distance_calculation
import social_distance_monitoring
import pandas as pd
from model_load import Model
from img_video_operations import ImageImport

"""
This module implements simple test for accuracy of social distance monitoring model
"""

if __name__ == "__main__":

    dataset = ImageImport(path='Images')
    test_dataset = dataset.image

    width, height = test_dataset[0].shape[:2]

    pts1 = ((78, 0), (560, 0), (0, height), (width, height))
    pts2 = ((0, 0), (width, 0), (0, height), (width, height))
    pt1_coord = (100, 210)
    pt2_coord = (580, 200)

    homography_matrix = homography.compute_homography(points1=pts1, points2=pts2)
    model_path = r"faster_rcnn_inception_v2_coco_2018_01_28\frozen_inference_graph.pb"
    model = Model(model_path)

    min_pixel_dist = distance_calculation.min_pixel_distance(obj1_coord=pt1_coord, obj2_coord=pt2_coord,
                                                             homography_matrix=homography_matrix,
                                                             object_real_dist=8, min_real_dist=1.5)

    sdm = social_distance_monitoring.SocialDistanceMonitoring(model=model, image_dataset=test_dataset,
                                                              homography_matrix=homography_matrix,
                                                              real_pixel_distance=min_pixel_dist)

    data = pd.read_csv('data.csv', header=0, sep=";")
    accuracy, number_of_people = sdm.find_monitoring_accuracy(data)

    print(f"Number of test cases: {number_of_people}")
    print("Monitoring accuracy: %.2f" % accuracy)
