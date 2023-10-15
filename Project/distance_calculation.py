# -*- coding: utf-8 -*-
"""
Script calculates the distance between detected figures
"""
import itertools
import math
import numpy as np


def min_pixel_distance(obj1_coord, obj2_coord, homography_matrix, object_real_dist, min_real_dist):
    """
    Function returns the minimum distance in pixels based on the real distance of two 
    fixed objects in the image.

    parameters
        objects_coord: <list> coordinates of the objects taken as reference
        homography_matrix <array/list> homography matrix calculated in previous stages  
        object_real_dist: <float> real distance between the objects
        min_real_dist: <float> social distance which should be maintain in real life
                                (threshold in the reality)
    """    
    obj1 = np.array(list(obj1_coord)+[1])
    obj2 = np.array(list(obj2_coord)+[1])

    #transforming the original files using the homography matrix
    obj1_trans = homography_matrix@obj1 / (homography_matrix@obj1)[-1] 
    obj2_trans = homography_matrix@obj2 / (homography_matrix@obj2)[-1]
    
    x_dist = obj1_trans[0]-obj2_trans[0]
    y_dist = obj1_trans[1]-obj2_trans[1]
    dist = math.sqrt(x_dist**2 + y_dist**2)

    # we calculate real distance of a single pixel

    dist_pixel_ratio = object_real_dist/dist
    min_pixel_dist = math.floor(min_real_dist/dist_pixel_ratio)
    
    return min_pixel_dist


def min_pixel_distance_auto(boxes_coordinates, homography_matrix, min_real_dist, avg_height=1.73):
    """
    Function returns the minimum distance in pixels calculated in the more automatic way.
    It takes the average height of the detected figures in the image (with filtering out 
    of the uncommon values) and calculates te ratio considering the human average height 
    in real life.

    parameters
        boxes_coordinates: <list> coordinates of the boxes of the detected figures
        homography_matrix <array/list> homography matrix calculated in previous stages  
        min_real_dist: <float> social distance which should be maintain in real life
                                (threshold in the reality)
        avg_height: average human height in real life
    """
    heights = []
    
    for i in boxes_coordinates:
        person_top = np.array(list([i[0], (i[1]+i[3])/2]) + [1])
        person_bottom = np.array(list([i[2], (i[1]+i[3])/2]) + [1])
        
        person_top_trans = homography_matrix@person_top
        person_bottom_trans = homography_matrix@person_bottom   
        
        heights.append(abs(person_top_trans[0]-person_bottom_trans[0]))
    
    # we filter out observations which are far from the mean - the diffence is higher 
    # than the standard deviation
    heights_arr = np.array(heights)
    heights_arr = heights_arr[np.where(abs(heights-np.mean(heights)) > np.std(heights))]

    if len(heights_arr) > 0:
        mean_heights = np.mean(heights_arr)
    else:
        mean_heights = np.median(heights_arr)
        
    # default average height is equal 1.73 as in Poland average height of woman and man is,
    # respectively, 1.65 and 1.80
        
    dist_pixel_ratio = avg_height/mean_heights

    min_pixel_dist = math.floor(min_real_dist/dist_pixel_ratio)    

    return min_pixel_dist


def social_distance(coordinates, min_distance):
    """
    Function calculates distance between detected people and classify whether the safe 
    distance is maintained. It returns number of social distance violation cases and
    coordinates of the objects which did not maintain the minimum distance.
    
    parameters:
        coordinates: <list> list of the coordinates of points indicating detected people
        min_distance: <integer> threshold of the distance that has to be maintained
                      (calculated in previous stage)
    """
    
    if len(coordinates) < 2:
        return None, None, None

    people_violated = set()
    violating_neighbor_list = [[] for _ in range(len(coordinates))]

    for i in list(itertools.combinations(range(len(coordinates)), 2)):
        
        #distance computation in the Euclidean metrics
        x_dist = coordinates[i[0]][0]-coordinates[i[1]][0]
        y_dist = coordinates[i[0]][1]-coordinates[i[1]][1]
        dist = math.sqrt(x_dist**2 + y_dist**2)
        
        if dist < min_distance:
            people_violated.add(tuple(coordinates[i[0]]))
            people_violated.add(tuple(coordinates[i[1]]))
            violating_neighbor_list[i[0]].append(i[1])
            violating_neighbor_list[i[1]].append(i[0])

    violation_cases = len(people_violated)

    return violation_cases, people_violated, violating_neighbor_list
