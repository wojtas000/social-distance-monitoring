# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from social_distance_monitoring import SocialDistanceMonitoring
from homography import compute_homography
from distance_calculation import min_pixel_distance, min_pixel_distance_auto
from model_load import Model
from img_video_operations import VideoImport, ImageImport

"""
Main module of social distance monitoring project - example with 
the application of the min_pixel_distance function from the
distance_calculation2 script 
"""


def main():
    # interactive part for the user"
    movie_or_photos = int(input("""Please choose if you want to import the video (with the .avi extension) or images to the model: 
                                \n Type the number corresponding to your selection: \n 1 - Video \n 2 - Images\n"""))

    if movie_or_photos == 1:
        path = str(input("Specify the path to the directory where the video is located:\n"))
        file_name = str(input("""Type the name of the file. If you won't type the name, the program will select the first relevant file from the directory: \n"""))
        file_imported = VideoImport(path=path, name=file_name)
        
        frames_num = int(input(f"""Enter the number of frames that you want to import to the model, there is a total of {file_imported.frames_number} frames"""))
        matrices_img = file_imported.make_frames(frames_num)
    
    elif movie_or_photos == 2:
        path = str(input("Specify the path to the directory where the image is located:\n"))
        file_name = str(input("""Type the name of the file. If you won't type the name, the program will select all relevant files from the directory: \n"""))
        file_imported = ImageImport(path=path, name=file_name)
        matrices_img = file_imported.image

    # points input for the Homography matrix calculations
    pts_original = input("Enter the coordinates of the four points chosen for the Homography matrix calculation:" +
                         "(different points should be separated by comma and "
                         "there should be a space between coordinates in each point)"
                         "\n e.g. 0 0, 0 5, 3 2, 3 4\n")
    
    pts_homography = input("Enter coordinates of these coordinated after the homography transformation:\n")

    pts_list = []
    for point in pts_original.split(','):
        pts_list.append((int(point.strip().split(' ')[0]), int(point.strip().split(' ')[1])))
        
    pts1 = tuple(pts_list)

    pts_list = []
    for point in pts_homography.split(','):
        pts_list.append((int(point.strip().split(' ')[0]), int(point.strip().split(' ')[1])))
        
    pts2 = tuple(pts_list)

    distance_measure_version = int(input("""Select the version which you want to apply in the calculation of the real distance of a pixel: \n 1 - automatic calculations considering the average height of people \n 2 - calculations with the consideration of two fixed points in the images or frames:\n"""))

    homography_matrix = compute_homography(points1=pts1, points2=pts2)

    print("Loading the model...")
    model_path = r"faster_rcnn_inception_v2_coco_2018_01_28\frozen_inference_graph.pb"
    model = Model(model_path)

    if distance_measure_version == 2:
        x1 = int(input("Enter the first coordinate of the first point: "))
        y1 = int(input("Enter the second coordinate of the first point: "))
        x2 = int(input("Enter the first coordinate of the second point: "))
        y2 = int(input("Enter the second coordinate of the second point: "))
        real_dist = float(input("Enter the real distance of two selected points: "))         
                 
        pt1_coord = (x1, y1)
        pt2_coord = (x2, y2)
        
        min_pixel_dist = min_pixel_distance(obj1_coord=pt1_coord, obj2_coord=pt2_coord,
                                            homography_matrix=homography_matrix,
                                            object_real_dist=real_dist, min_real_dist=1.5)
        print("Minimum distance in pixels: ", min_pixel_dist)
        
        # we define the object created for social distance monitoring (it is a kind of a summary
        # of the image analysis)

        sdm = SocialDistanceMonitoring(model=model, image_dataset=matrices_img,
                                       homography_matrix=homography_matrix,
                                       real_pixel_distance=min_pixel_dist)
        
        # Printing accuracy of social distance monitoring

        data = pd.read_csv('data.csv', header=0, sep=";")
        accuracy = sdm.find_monitoring_accuracy(data)
        print("Monitoring accuracy: %.2f" % accuracy)
    
    elif distance_measure_version == 1:
        sdm = SocialDistanceMonitoring(model=model, image_dataset=matrices_img,
                                       homography_matrix=homography_matrix,
                                       real_pixel_distance=0)

    iter_num = 0
    
    for image in matrices_img:

        # loading the image

        copy_image = image.copy()
        
        # on the image we place points the distance between which we consider as reference

        if distance_measure_version == 2:
            cv2.circle(img=image, center=pt1_coord, radius=8,
                       color=(0, 0, 255), thickness=5)
    
            cv2.circle(img=image, center=pt2_coord, radius=8,
                       color=(0, 0, 255), thickness=5)

        # we run the model on the image and generate boxes of people who do not maintain
        # minimum distance

        _, scores, classes = sdm.model.predict(image)
        boxes = sdm.model.generate_boxes(image)
        
        if iter_num == 0:
            if distance_measure_version == 1:
                min_pixel_dist = min_pixel_distance_auto(boxes_coordinates=boxes, homography_matrix=homography_matrix, 
                                                         min_real_dist=1.5, avg_height=1.73)
                print("Minimum distance in pixels: ", min_pixel_dist)
                
                sdm.real_pixel_distance = min_pixel_dist
                
                # Printing accuracy of social distance monitoring
                data = pd.read_csv('data.csv', header=0, sep=";")
                accuracy, number_of_people = sdm.find_monitoring_accuracy(data)
                print("Monitoring accuracy: %.2f" % accuracy)
                print(f"Number of test cases: {number_of_people}")

        # we place the boxes on the image

        sdm.modify_image(copy_image, boxes)
        
        iter_num += 1

        # generating the plot of the source image and the image with the results

        fig = sdm.bird_eye_view(boxes)
        plt.show()

        figure = plt.figure(figsize=(12, 6))
        subplot1 = figure.add_subplot(1, 2, 1)
        subplot1.title.set_text(f"Image seq_00000{iter_num}.jpg")
        subplot1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        subplot2 = figure.add_subplot(1, 2, 2)
        subplot2.title.set_text("Image with figures detected")
        subplot2.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
        plt.show()
        

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
